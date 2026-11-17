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
"""Dynamo-traceable sequence/CFG-parallel primitives for LTX-2 / LTX-2.3 DiT."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

_V2A_SP_PATCHED = "_tpu_v2a_sp_patched"
_PERTURBATION_SYM_PATCHED = "_tpu_perturbation_sym_patched"


def _functional_all_to_all_4d(
    input_t: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group: dist.ProcessGroup | None = None,
    use_sync: bool = False,
) -> torch.Tensor:
    """Functional ``all_to_all_4D`` lowering cleanly to ``_c10d_functional``."""
    del use_sync
    seq_world_size = dist.get_world_size(group)
    if seq_world_size <= 1:
        return input_t

    if scatter_idx == 2 and gather_idx == 1:
        bs, shard_seqlen, hc, hs = input_t.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size
        x = (
            input_t.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )
        out = funcol.all_to_all_single(x, None, None, group)
        return (
            out.reshape(seqlen, bs, shard_hc, hs)
            .transpose(0, 1)
            .contiguous()
            .reshape(bs, seqlen, shard_hc, hs)
        )

    if scatter_idx == 1 and gather_idx == 2:
        bs, seqlen, shard_hc, hs = input_t.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        x = (
            input_t.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .permute(1, 3, 2, 0, 4)
            .contiguous()
        )
        out = funcol.all_to_all_single(x, None, None, group)
        return (
            out.reshape(hc, shard_seqlen, bs, hs)
            .permute(2, 1, 0, 3)
            .contiguous()
            .reshape(bs, shard_seqlen, hc, hs)
        )

    raise RuntimeError(
        f"Invalid (scatter_idx, gather_idx)=({scatter_idx}, {gather_idx}); "
        "expected (2, 1) or (1, 2)"
    )


def _patch_ltx2_traceable_collectives() -> None:
    """Make Ulysses and LTX-2 video-to-audio SP collectives Dynamo-traceable."""
    from vllm_omni.diffusion.attention.parallel import ulysses as ulysses_mod
    from vllm_omni.diffusion.distributed import comm as comm_mod
    from vllm_omni.diffusion.models.ltx2 import ltx2_sequence_parallel as ltx2_sp_mod

    v2a_cls = ltx2_sp_mod.LTX2VideoToAudioParallelAttention
    if getattr(v2a_cls.pre_attention, _V2A_SP_PATCHED, False):
        return

    orig_seq_a2a = comm_mod.SeqAllToAll4D

    class _TraceableSeqAllToAll4D:
        @staticmethod
        def apply(
            group: dist.ProcessGroup,
            input_t: torch.Tensor,
            scatter_idx: int,
            gather_idx: int,
            use_sync: bool = False,
        ) -> torch.Tensor:
            if torch.compiler.is_compiling():
                return _functional_all_to_all_4d(
                    input_t,
                    scatter_idx=scatter_idx,
                    gather_idx=gather_idx,
                    group=group,
                    use_sync=use_sync,
                )
            return orig_seq_a2a.apply(group, input_t, scatter_idx, gather_idx, use_sync)

    comm_mod.SeqAllToAll4D = _TraceableSeqAllToAll4D  # type: ignore[assignment]
    ulysses_mod.SeqAllToAll4D = _TraceableSeqAllToAll4D  # type: ignore[assignment]
    ltx2_sp_mod.SeqAllToAll4D = _TraceableSeqAllToAll4D  # type: ignore[assignment]

    orig_pre = v2a_cls.pre_attention
    orig_post = v2a_cls.post_attention

    def _traceable_v2a_pre(
        self: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any,
    ) -> Any:
        if not torch.compiler.is_compiling():
            return orig_pre(self, query, key, value, attn_metadata)
        world_size = self._sp_group.ulysses_world_size
        query_heads = int(query.shape[2])
        heads_per_rank = query_heads // world_size
        head_idx = getattr(self, "_tpu_head_idx", None)
        if head_idx is None:
            raise RuntimeError(
                "LTX2VideoToAudioParallelAttention._tpu_head_idx must be initialized "
                "outside torch.compile before compiled execution."
            )
        if head_idx.shape[0] != heads_per_rank:
            raise ValueError(
                f"Expected {heads_per_rank} heads in _tpu_head_idx, "
                f"got {head_idx.shape[0]}"
            )
        q_local = torch.index_select(query, 2, head_idx)
        k_full = _functional_all_to_all_4d(key, 2, 1, group=self._ulysses_pg)
        v_full = _functional_all_to_all_4d(value, 2, 1, group=self._ulysses_pg)
        ctx = ltx2_sp_mod._VideoToAudioContext(
            name=self.name,
            ulysses_pg=self._ulysses_pg,
            use_sync=self._use_sync,
        )
        return q_local, k_full, v_full, attn_metadata, ctx

    def _traceable_v2a_post(
        self: Any,
        attn_output: torch.Tensor,
        ctx: Any,
    ) -> torch.Tensor:
        if not torch.compiler.is_compiling():
            return orig_post(self, attn_output, ctx)
        local_heads_first = attn_output.permute(2, 0, 1, 3).contiguous()
        gathered = funcol.all_gather_tensor(local_heads_first, 0, ctx.ulysses_pg)
        return gathered.permute(1, 2, 0, 3).contiguous()

    setattr(_traceable_v2a_pre, _V2A_SP_PATCHED, True)
    v2a_cls.pre_attention = _traceable_v2a_pre
    v2a_cls.post_attention = _traceable_v2a_post


def _patch_ltx2_symmetric_perturbation_masks() -> None:
    """Keep perturbation masks structurally identical across all CFG-parallel ranks."""
    from vllm_omni.diffusion.models.ltx2 import ltx2_guidance

    exec_cls = ltx2_guidance.LTXGuidanceExecutor
    if getattr(exec_cls.predict_parallel_guidance, _PERTURBATION_SYM_PATCHED, False):
        return

    orig_build_perturbation = ltx2_guidance.build_perturbation_kwargs
    orig_predict_parallel = exec_cls.predict_parallel_guidance

    def _sym_predict_parallel_guidance(
        self: Any,
        pipeline: Any,
        plan: Any,
        index: int,
        timestep: torch.Tensor,
        state: Any,
        forward_ctx: Any,
        denoise_ctx: Any,
        preserve_positive_velocity: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def _sym_build_perturbation_kwargs(
            local_plan: Any,
            batch_size: int,
            reference: torch.Tensor,
        ) -> dict[str, Any]:
            kw = orig_build_perturbation(local_plan, batch_size, reference)
            local_total = len(local_plan.passes) * batch_size
            ones: torch.Tensor | None = None

            def _get_ones() -> torch.Tensor:
                nonlocal ones
                if ones is None:
                    ones = reference.new_ones((local_total, 1, 1))
                    synchronize_tensors(ones, wait=False)
                return ones

            if "ptb" in plan.names and "ptb" not in local_plan.names:
                if plan.spec.video.do_stg:
                    kw["video_self_attention_mask"] = _get_ones()
                    kw["video_self_attention_blocks"] = plan.spec.video.stg_blocks
                if plan.spec.audio.do_stg:
                    kw["audio_self_attention_mask"] = _get_ones()
                    kw["audio_self_attention_blocks"] = plan.spec.audio.stg_blocks
            if "mod" in plan.names and "mod" not in local_plan.names:
                kw["a2v_cross_attention_mask"] = _get_ones()
                kw["v2a_cross_attention_mask"] = _get_ones()
            return kw

        ltx2_guidance.build_perturbation_kwargs = _sym_build_perturbation_kwargs
        try:
            return orig_predict_parallel(
                self,
                pipeline,
                plan,
                index,
                timestep,
                state,
                forward_ctx,
                denoise_ctx,
                preserve_positive_velocity=preserve_positive_velocity,
            )
        finally:
            ltx2_guidance.build_perturbation_kwargs = orig_build_perturbation

    setattr(_sym_predict_parallel_guidance, _PERTURBATION_SYM_PATCHED, True)
    exec_cls.predict_parallel_guidance = _sym_predict_parallel_guidance
