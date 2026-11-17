# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import SimpleNamespace

import pytest
import torch
from vllm_omni.diffusion.distributed import comm, parallel_state
from vllm_omni.diffusion.models.ltx2 import ltx2_guidance, ltx2_transformer

from vllm_torchtpu.omni import ltx2_compile

pytestmark = pytest.mark.omni

_UPSTREAM_BUILD_PERTURBATION = ltx2_guidance.build_perturbation_kwargs
_UPSTREAM_PREDICT_PARALLEL = ltx2_guidance.LTXGuidanceExecutor.predict_parallel_guidance
_UPSTREAM_ATTN_PROC_CALL = ltx2_transformer.LTX2AudioVideoAttnProcessor.__call__


@pytest.fixture
def sync_log(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ltx2_compile,
        "synchronize_tensors",
        lambda tensors, wait=False: calls.append((tensors, wait)),
    )
    return calls


def test_functional_all_to_all_4d_matches_upstream(monkeypatch):
    world_size = 4
    monkeypatch.setattr(
        ltx2_compile.dist, "get_world_size", lambda group=None: world_size
    )
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: world_size
    )

    def _run_world_funcol(rank_inputs, scatter_idx, gather_idx):
        sends = []
        monkeypatch.setattr(
            ltx2_compile.funcol,
            "all_to_all_single",
            lambda x, out_s, in_s, group: sends.append(x.clone())
            or torch.zeros_like(x),
        )
        for r in range(world_size):
            ltx2_compile._functional_all_to_all_4d(
                rank_inputs[r], scatter_idx, gather_idx, group="pg"
            )
        chunked = torch.stack(
            [torch.stack(s.chunk(world_size, dim=0), dim=0) for s in sends], dim=0
        )
        transposed = chunked.transpose(0, 1).contiguous()
        outputs = []
        for r in range(world_size):
            recv_flat = transposed[r].reshape_as(sends[r])
            monkeypatch.setattr(
                ltx2_compile.funcol,
                "all_to_all_single",
                lambda x, out_s, in_s, group, rf=recv_flat: rf.clone(),
            )
            outputs.append(
                ltx2_compile._functional_all_to_all_4d(
                    rank_inputs[r], scatter_idx, gather_idx, group="pg"
                )
            )
        return outputs

    def _run_world_eager(rank_inputs, scatter_idx, gather_idx):
        sends = []

        def collect(recv_buf, send_buf, group=None):
            del group
            sends.append(send_buf.clone())
            recv_buf.zero_()

        monkeypatch.setattr(torch.distributed, "all_to_all_single", collect)
        for r in range(world_size):
            comm.all_to_all_4D(rank_inputs[r], scatter_idx, gather_idx, group="pg")
        transposed = torch.stack(sends, dim=0).transpose(0, 1).contiguous()
        outputs = []
        for r in range(world_size):
            r_recv = transposed[r]
            monkeypatch.setattr(
                torch.distributed,
                "all_to_all_single",
                lambda recv_buf, send_buf, group=None, rr=r_recv: recv_buf.copy_(rr),
            )
            outputs.append(
                comm.all_to_all_4D(rank_inputs[r], scatter_idx, gather_idx, group="pg")
            )
        return outputs

    torch.manual_seed(0)
    in_21 = [torch.randn(2, 3, 8, 4) for _ in range(world_size)]
    for g, w in zip(
        _run_world_funcol(in_21, 2, 1), _run_world_eager(in_21, 2, 1), strict=True
    ):
        torch.testing.assert_close(g, w, rtol=0, atol=0)

    in_12 = [torch.randn(2, 12, 2, 4) for _ in range(world_size)]
    for g, w in zip(
        _run_world_funcol(in_12, 1, 2), _run_world_eager(in_12, 1, 2), strict=True
    ):
        torch.testing.assert_close(g, w, rtol=0, atol=0)


def test_symmetric_perturbation_masks_across_cfg_ranks(monkeypatch, sync_log):
    monkeypatch.setattr(
        ltx2_guidance, "build_perturbation_kwargs", lambda lp, bs, r: {}
    )
    captured_kwargs = []
    monkeypatch.setattr(
        ltx2_guidance.LTXGuidanceExecutor,
        "predict_parallel_guidance",
        lambda self, *args, **kwargs: (
            captured_kwargs.append(
                ltx2_guidance.build_perturbation_kwargs(
                    SimpleNamespace(names=("uncond", "cond"), passes=(1, 2)),
                    1,
                    torch.zeros(2, 4, 8),
                )
            )
            or (torch.zeros(1), torch.zeros(1))
        ),
    )

    ltx2_compile._patch_ltx2_symmetric_perturbation_masks()
    plan = SimpleNamespace(
        names=("uncond", "cond", "ptb", "mod"),
        spec=SimpleNamespace(
            video=SimpleNamespace(do_stg=True, stg_blocks=[28]),
            audio=SimpleNamespace(do_stg=True, stg_blocks=[28]),
        ),
    )
    executor = ltx2_guidance.LTXGuidanceExecutor()
    executor.predict_parallel_guidance(
        None, plan, 0, torch.tensor(1.0), None, None, None
    )
    kw = captured_kwargs[0]
    assert kw["video_self_attention_blocks"] == [28]
    assert kw["audio_self_attention_blocks"] == [28]
    for key in (
        "video_self_attention_mask",
        "audio_self_attention_mask",
        "a2v_cross_attention_mask",
        "v2a_cross_attention_mask",
    ):
        assert kw[key].shape == (2, 1, 1)
        assert torch.all(kw[key] == 1.0)


def test_ensure_zero_offset_and_rope_cache(sync_log):
    full = torch.randn(2, 1, 8, 16)
    assert ltx2_compile._ensure_zero_offset(full) is full
    assert len(sync_log) == 0

    cache = {}
    view0 = full[:, :, 0:4, :]
    view1 = full[:, :, 4:8, :]
    m0 = ltx2_compile._ensure_zero_offset(view0, cache)
    assert m0.is_contiguous() and m0.storage_offset() == 0
    torch.testing.assert_close(m0, view0, rtol=0, atol=0)
    assert ltx2_compile._ensure_zero_offset(full[:, :, 0:4, :], cache) is m0
    assert len(sync_log) == 1

    m1 = ltx2_compile._ensure_zero_offset(view1, cache)
    assert m1.storage_offset() == 0
    torch.testing.assert_close(m1, view1, rtol=0, atol=0)
    assert len(sync_log) == 2


def test_shared_block_compile_binds_one_graph_across_blocks(monkeypatch, sync_log):
    monkeypatch.setattr(parallel_state, "get_sequence_parallel_world_size", lambda: 1)

    compile_calls = []

    def fake_compile(backend, fullgraph, dynamic):
        compile_calls.append((backend, fullgraph, dynamic))
        return lambda fn: fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    class _ToyBlock(torch.nn.Module):
        def __init__(self, scale: float) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.lin.weight.copy_(torch.eye(4) * scale)

        def forward(self, hs, ahs, ehs, aehs, **kwargs):
            del ehs, aehs, kwargs
            return self.lin(hs), ahs + 1.0

    class _ToyTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                num_attention_heads=2,
                attention_head_dim=4,
                audio_num_attention_heads=2,
                audio_attention_head_dim=4,
            )
            self.transformer_blocks = torch.nn.ModuleList(
                [_ToyBlock(2.0), _ToyBlock(3.0), _ToyBlock(5.0)]
            )

    model = _ToyTransformer()
    ltx2_compile._install_shared_block_compile(model)
    ltx2_compile._install_shared_block_compile(model)  # idempotent
    assert compile_calls == [("tpu", True, False)]

    x = torch.ones(1, 2, 4)
    a = torch.zeros(1, 2, 4)
    e = torch.zeros(1, 2, 4)
    for blk in model.transformer_blocks:
        x, a = blk(x, a, e, e)
    torch.testing.assert_close(x, torch.full((1, 2, 4), 30.0))
    torch.testing.assert_close(a, torch.full((1, 2, 4), 3.0))


def test_fresh_ltx2_forward_flushes_outputs(monkeypatch, sync_log):
    monkeypatch.setattr(ltx2_compile, "_patch_ltx2_traceable_collectives", lambda: None)
    monkeypatch.setattr(
        ltx2_compile, "_patch_ltx2_symmetric_perturbation_masks", lambda: None
    )
    monkeypatch.setattr(
        ltx2_transformer.LTX2VideoTransformer3DModel,
        "forward",
        lambda self, *args, **kwargs: (
            torch.ones(1, 4),
            torch.ones(1, 2),
        ),
    )

    ltx2_compile._activate_ltx2_compile_runtime_hooks()
    dummy = object.__new__(ltx2_transformer.LTX2VideoTransformer3DModel)
    got = ltx2_transformer.LTX2VideoTransformer3DModel.forward(
        dummy,
        torch.zeros(1, 4),
        torch.zeros(1, 2),
        torch.zeros(1, 4),
        torch.zeros(1, 2),
        torch.tensor([500.0]),
        torch.tensor([500.0]),
        sigma=torch.tensor([0.5]),
    )
    assert len(got) == 2
    assert len(sync_log) == 2
    assert len(sync_log[0][0]) == 3 and sync_log[0][1] is False
    assert len(sync_log[1][0]) == 2 and sync_log[1][1] is False


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_gates", [True, False])
def test_mxu_gated_attention_matches_upstream(monkeypatch, dtype, use_gates):
    monkeypatch.setattr(
        ltx2_transformer.LTX2AudioVideoAttnProcessor,
        "__call__",
        _UPSTREAM_ATTN_PROC_CALL,
    )
    ltx2_compile._patch_ltx2_mxu_gated_attention()
    ltx2_compile._patch_ltx2_mxu_gated_attention()  # idempotent

    torch.manual_seed(0)
    heads, head_dim = 4, 8
    dim = heads * head_dim
    gate_proj = torch.nn.Linear(dim, heads).to(dtype=dtype) if use_gates else None
    q_proj = torch.nn.Linear(dim, dim, bias=False).to(dtype=dtype)
    k_proj = torch.nn.Linear(dim, dim, bias=False).to(dtype=dtype)
    v_proj = torch.nn.Linear(dim, dim, bias=False).to(dtype=dtype)
    out_proj = torch.nn.Linear(dim, dim, bias=False).to(dtype=dtype)

    class _StubAttn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = heads
            self.head_dim = head_dim
            self.rope_type = "split"
            self.to_gate_logits = gate_proj
            self.norm_q = torch.nn.Identity()
            self.norm_k = torch.nn.Identity()
            self.to_out = torch.nn.ModuleList([out_proj, torch.nn.Identity()])
            self.attn = lambda q, k, v, meta=None: (q + k + v) * 0.25

    proc = ltx2_transformer.LTX2AudioVideoAttnProcessor()
    monkeypatch.setattr(
        proc,
        "_prepare_attention_mask",
        lambda **kw: kw["attention_mask"],
    )
    monkeypatch.setattr(
        proc,
        "_project_qkv",
        lambda attn, hidden_states, encoder_hidden_states, is_self_attention: (
            q_proj(hidden_states),
            k_proj(encoder_hidden_states),
            v_proj(encoder_hidden_states),
        ),
    )

    attn_mod = _StubAttn()
    hs = torch.randn(2, 6, dim, dtype=dtype)
    pmask = torch.tensor([[[1.0]], [[0.0]]], dtype=dtype)

    got = proc(attn_mod, hs, perturbation_mask=pmask)
    want = _UPSTREAM_ATTN_PROC_CALL(proc, attn_mod, hs, perturbation_mask=pmask)
    assert got.dtype == dtype
    # One 1.0 entry per column in expand_mat makes bf16 indicator matmul bit-exact.
    torch.testing.assert_close(got, want, rtol=0, atol=0)
