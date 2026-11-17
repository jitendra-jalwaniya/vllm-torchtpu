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
from vllm_omni.diffusion.distributed import comm
from vllm_omni.diffusion.models.ltx2 import ltx2_guidance

from vllm_torchtpu.omni import ltx2_compile

pytestmark = pytest.mark.omni

_UPSTREAM_BUILD_PERTURBATION = ltx2_guidance.build_perturbation_kwargs
_UPSTREAM_PREDICT_PARALLEL = ltx2_guidance.LTXGuidanceExecutor.predict_parallel_guidance


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
