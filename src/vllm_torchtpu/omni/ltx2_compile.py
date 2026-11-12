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
"""TPU torch.compile + layout/sync optimizations for LTX-2 / LTX-2.3 DiT.

Compiles ONE shared ``LTX2VideoTransformerBlock`` graph across all 48 blocks
with ``torch.compile(backend="tpu", fullgraph=True, dynamic=False)``, removing
the ~263 ms/step eager Python/ATen dispatch floor while keeping cold compile
latency low (reusing a single compiled callable across all homogeneous blocks).

Additional toggleable TPU optimizations on the stack branch:
  * ``VLLM_TORCHTPU_LTX2_ASYNC_STEP=1`` (default): replaces CUDA-only blocking
    ``current_omni_platform.synchronize()`` in
    ``LTXRuntime._synchronize_guidance_parallel_step_output`` with non-blocking
    ``synchronize_tensors(..., wait=False)``, eliminating 40 per-step host-device
    pipeline drains.
  * ``VLLM_TORCHTPU_LTX2_MXU_GATE=1`` (default): expands LTX-2.3 per-head gate
    scalars ``(B, T, H) -> (B, T, H * D)`` via a bit-exact ``(H, H * D)`` MXU
    indicator matmul instead of ``unflatten(2, (H, D)) * gates.unsqueeze(-1)``,
    eliminating 12 synchronous 4D relayout copies per block.
"""

from __future__ import annotations

import functools
import os
import types
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.nn.utils import stateless

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.omni.rope import _first_half_lanes
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

_LTX2_COMPILE_PATCHED = "_tpu_ltx2_compile_patched"
_V2A_SP_PATCHED = "_tpu_v2a_sp_patched"
_PERTURBATION_SYM_PATCHED = "_tpu_perturbation_sym_patched"
_ASYNC_STEP_PATCHED = "_tpu_ltx2_async_step_patched"
_MXU_GATE_PATCHED = "_tpu_ltx2_mxu_gate_patched"
_V2A_HEAD_IDX_CACHE: dict[int, torch.Tensor] = {}
_ROPE_KEYS = frozenset(
    {
        "video_rotary_emb",
        "audio_rotary_emb",
        "ca_video_rotary_emb",
        "ca_audio_rotary_emb",
    }
)


def _build_gate_expand_matrix(
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cols = torch.arange(heads * head_dim, device=device) // head_dim
    rows = torch.arange(heads, device=device).unsqueeze(1)
    return (rows == cols.unsqueeze(0)).to(dtype=dtype)


@functools.cache
def _cached_gate_expand_matrix(
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return _build_gate_expand_matrix(heads, head_dim, dtype, device)


def _patch_ltx2_async_guidance_step() -> None:
    """Replace CUDA-only per-step device synchronize() with non-blocking flush."""
    if os.environ.get("VLLM_TORCHTPU_LTX2_ASYNC_STEP", "1") == "0":
        return

    from vllm_omni.diffusion.distributed.parallel_state import (
        get_guidance_parallel_world_size,
    )
    from vllm_omni.diffusion.models.ltx2.ltx2_runtime import LTXRuntime

    if getattr(
        LTXRuntime._synchronize_guidance_parallel_step_output,
        _ASYNC_STEP_PATCHED,
        False,
    ):
        return

    def _async_guidance_step_output(
        self: Any,
        latents: tuple[torch.Tensor, torch.Tensor],
        guidance_parallel_ready: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del self
        if not (guidance_parallel_ready and get_guidance_parallel_world_size() > 1):
            return latents
        latents = tuple(tensor.contiguous() for tensor in latents)
        synchronize_tensors(list(latents), wait=False)
        return latents

    setattr(_async_guidance_step_output, _ASYNC_STEP_PATCHED, True)
    LTXRuntime._synchronize_guidance_parallel_step_output = _async_guidance_step_output
    logger.info(
        "Applied TPU patch: non-blocking async guidance-parallel step boundary."
    )


def _patch_ltx2_mxu_gated_attention() -> None:
    """Expand LTX-2.3 per-head attention gates on MXU without 4D relayout copies."""
    if os.environ.get("VLLM_TORCHTPU_LTX2_MXU_GATE", "1") == "0":
        return

    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.models.ltx2 import ltx2_transformer

    proc_cls = ltx2_transformer.LTX2AudioVideoAttnProcessor
    if getattr(proc_cls.__call__, _MXU_GATE_PATCHED, False):
        return

    def _mxu_gated_attn_call(
        self: Any,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        perturbation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_self_attention = encoder_hidden_states is None
        batch_size, sequence_length, _ = (
            hidden_states.shape if is_self_attention else encoder_hidden_states.shape
        )

        attention_mask = self._prepare_attention_mask(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

        gate_logits: torch.Tensor | None = None
        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(hidden_states)
            if isinstance(gate_logits, tuple):
                gate_logits = gate_logits[0]

        if is_self_attention:
            encoder_hidden_states = hidden_states

        query, key, value = self._project_qkv(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            is_self_attention=is_self_attention,
        )

        query = attn.norm_q(query).to(dtype=value.dtype)
        key = attn.norm_k(key).to(dtype=value.dtype)

        if query_rotary_emb is not None:
            query_rotary_emb = self._slice_rope_for_tp(query_rotary_emb, attn)
            if key_rotary_emb is not None:
                key_rotary_emb = self._slice_rope_for_tp(key_rotary_emb, attn)
            if attn.rope_type == "interleaved":
                query = ltx2_transformer.apply_interleaved_rotary_emb(
                    query, query_rotary_emb
                )
                key = ltx2_transformer.apply_interleaved_rotary_emb(
                    key,
                    key_rotary_emb if key_rotary_emb is not None else query_rotary_emb,
                )
            elif attn.rope_type == "split":
                query = ltx2_transformer.apply_split_rotary_emb(
                    query, query_rotary_emb, head_dim=attn.head_dim
                )
                key = ltx2_transformer.apply_split_rotary_emb(
                    key,
                    key_rotary_emb if key_rotary_emb is not None else query_rotary_emb,
                    head_dim=attn.head_dim,
                )

        query = query.to(dtype=value.dtype)
        key = key.to(dtype=value.dtype)

        query = query.unflatten(2, (attn.heads, attn.head_dim))
        key = key.unflatten(2, (attn.heads, attn.head_dim))
        value_heads = value.unflatten(2, (attn.heads, attn.head_dim))

        attn_metadata = (
            AttentionMetadata(attn_mask=attention_mask)
            if attention_mask is not None
            else None
        )
        hidden_states = attn.attn(query, key, value_heads, attn_metadata)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        if perturbation_mask is not None:
            hidden_states = hidden_states * perturbation_mask + value * (
                1 - perturbation_mask
            )

        if gate_logits is not None:
            gates = 2.0 * torch.sigmoid(gate_logits)
            if hidden_states.dtype == torch.bfloat16 and gates.dtype == torch.bfloat16:
                expand_mat = (
                    _build_gate_expand_matrix(
                        attn.heads,
                        attn.head_dim,
                        gates.dtype,
                        hidden_states.device,
                    )
                    if torch.compiler.is_compiling()
                    else _cached_gate_expand_matrix(
                        attn.heads,
                        attn.head_dim,
                        gates.dtype,
                        hidden_states.device,
                    )
                )
                hidden_states = hidden_states * torch.matmul(gates, expand_mat)
            else:
                hidden_states = (
                    hidden_states.unflatten(2, (attn.heads, attn.head_dim))
                    * gates.unsqueeze(-1)
                ).flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    setattr(_mxu_gated_attn_call, _MXU_GATE_PATCHED, True)
    proc_cls.__call__ = _mxu_gated_attn_call
    logger.info("Applied TPU patch: MXU channel-expanded LTX-2.3 gated attention.")


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
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )
        out = funcol.all_to_all_single(x, None, None, group)
        return (
            out.reshape(hc, shard_seqlen, bs, hs)
            .transpose(0, 2)
            .contiguous()
            .reshape(bs, shard_seqlen, hc, hs)
        )

    raise RuntimeError(
        f"scatter_idx ({scatter_idx}) and gather_idx ({gather_idx}) must be (2, 1) or (1, 2)"
    )


def _ensure_zero_offset(
    val: Any,
    rope_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> Any:
    """Materialize strided or non-zero-offset views before compiled block entry."""
    if isinstance(val, torch.Tensor):
        if val.is_contiguous() and val.storage_offset() == 0:
            return val
        if rope_cache is not None:
            ptr = val.untyped_storage().data_ptr()
            if ptr != 0:
                key = (
                    ptr,
                    val.storage_offset(),
                    tuple(val.shape),
                    val.stride(),
                    val.dtype,
                    val.device,
                )
                entry = rope_cache.get(key)
                if entry is not None and entry[0].untyped_storage().data_ptr() == ptr:
                    return entry[1]
                mat = torch.empty(val.shape, dtype=val.dtype, device=val.device)
                mat.copy_(val)
                synchronize_tensors(mat, wait=False)
                if len(rope_cache) >= 16:
                    rope_cache.clear()
                rope_cache[key] = (val, mat)
                return mat
        mat = torch.empty(val.shape, dtype=val.dtype, device=val.device)
        mat.copy_(val)
        synchronize_tensors(mat, wait=False)
        return mat
    if isinstance(val, tuple):
        return tuple(_ensure_zero_offset(x, rope_cache) for x in val)
    return val


def _patch_ltx2_traceable_collectives() -> None:
    """Make Ulysses and LTX2 video-to-audio SP collectives Dynamo-traceable."""
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
            return orig_seq_a2a.apply(
                group, input_t, scatter_idx, gather_idx, use_sync
            )

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
            head_idx = _V2A_HEAD_IDX_CACHE.get(heads_per_rank)
        if head_idx is None:
            raise RuntimeError(
                "LTX2VideoToAudioParallelAttention._tpu_head_idx must be initialized "
                "outside torch.compile before compiled execution."
            )
        if head_idx.shape[0] != heads_per_rank:
            raise ValueError(
                f"Expected _tpu_head_idx of length {heads_per_rank}, got {head_idx.shape[0]}"
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


def _install_shared_block_compile(transformer: torch.nn.Module) -> None:
    """Compile ONE shared block function and bind all 48 LTX2 blocks to it."""
    blocks = getattr(transformer, "transformer_blocks", None)
    if not blocks or len(blocks) == 0:
        return
    if getattr(transformer, "_tpu_shared_blocks_compiled", False):
        return

    from vllm_omni.diffusion.distributed.parallel_state import (
        get_sequence_parallel_world_size,
        get_sp_group,
    )

    device = next(transformer.parameters()).device
    cfg = transformer.config
    inner_dim = cfg.num_attention_heads * cfg.attention_head_dim
    audio_inner_dim = cfg.audio_num_attention_heads * cfg.audio_attention_head_dim
    _first_half_lanes(inner_dim, cfg.attention_head_dim, device)
    _first_half_lanes(audio_inner_dim, cfg.audio_attention_head_dim, device)

    sp_world = get_sequence_parallel_world_size()
    if sp_world > 1:
        sp_group = get_sp_group()
        urank = sp_group.ulysses_rank
        uworld = sp_group.ulysses_world_size
        heads_per_rank = cfg.audio_num_attention_heads // uworld
        start = urank * heads_per_rank
        head_idx = torch.arange(
            start, start + heads_per_rank, dtype=torch.int32, device=device
        )
        synchronize_tensors(head_idx, wait=False)
        _V2A_HEAD_IDX_CACHE[heads_per_rank] = head_idx
        from vllm_omni.diffusion.models.ltx2.ltx2_sequence_parallel import (
            LTX2VideoToAudioParallelAttention,
        )

        for blk in blocks:
            v2a_attn = getattr(blk, "video_to_audio_attn", None)
            inner_attn = getattr(v2a_attn, "attn", None)
            if (
                inner_attn is not None
                and getattr(inner_attn, "ltx_sp_mode", None) == "video_to_audio"
            ):
                if getattr(inner_attn, "_video_to_audio_strategy", None) is None:
                    inner_attn._video_to_audio_strategy = (
                        LTX2VideoToAudioParallelAttention(
                            sp_group,
                            use_sync=inner_attn.use_sync,
                        )
                    )
                inner_attn._video_to_audio_strategy._tpu_head_idx = head_idx

    template_block = blocks[0]
    orig_block_forward = type(template_block).forward
    rope_view_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _compiled_block_step(
        params_and_buffers: dict[str, torch.Tensor],
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        block_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with stateless._reparametrize_module(template_block, params_and_buffers):
            return orig_block_forward(
                template_block,
                hidden_states,
                audio_hidden_states,
                encoder_hidden_states,
                audio_encoder_hidden_states,
                **block_kwargs,
            )

    for blk in blocks:
        state_dict_tensors: dict[str, torch.Tensor] = {}
        for k, p in blk.named_parameters():
            state_dict_tensors[k] = p.detach()
        for k, b in blk.named_buffers():
            state_dict_tensors[k] = b.detach()

        def _make_bound_forward(
            pb: dict[str, torch.Tensor],
        ):
            @functools.wraps(orig_block_forward)
            def _bound_forward(
                _self: Any,
                hidden_states: torch.Tensor,
                audio_hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                audio_encoder_hidden_states: torch.Tensor,
                **kwargs: Any,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                hidden_states = _ensure_zero_offset(hidden_states)
                audio_hidden_states = _ensure_zero_offset(audio_hidden_states)
                encoder_hidden_states = _ensure_zero_offset(encoder_hidden_states)
                audio_encoder_hidden_states = _ensure_zero_offset(
                    audio_encoder_hidden_states
                )
                clean_kwargs = {
                    k: _ensure_zero_offset(
                        v, rope_view_cache if k in _ROPE_KEYS else None
                    )
                    for k, v in kwargs.items()
                }
                return _compiled_block_step(
                    pb,
                    hidden_states,
                    audio_hidden_states,
                    encoder_hidden_states,
                    audio_encoder_hidden_states,
                    clean_kwargs,
                )

            return _bound_forward

        blk.forward = types.MethodType(_make_bound_forward(state_dict_tensors), blk)  # type: ignore[method-assign]

    transformer._tpu_shared_blocks_compiled = True  # type: ignore[attr-defined]
    logger.info(
        "Installed shared torch.compile(backend='tpu', fullgraph=True, dynamic=False) "
        "callable across %d LTX2VideoTransformerBlock modules.",
        len(blocks),
    )


def _activate_ltx2_compile_runtime_hooks() -> None:
    """Install traceable collectives, symmetric guidance masks, and fresh-timestep forward."""
    from vllm_omni.diffusion.models.ltx2 import ltx2_transformer

    model_cls = ltx2_transformer.LTX2VideoTransformer3DModel
    if getattr(model_cls.forward, _LTX2_COMPILE_PATCHED, False):
        return

    _patch_ltx2_traceable_collectives()
    _patch_ltx2_symmetric_perturbation_masks()

    orig_model_forward = model_cls.forward

    @functools.wraps(orig_model_forward)
    def _fresh_ltx2_forward(
        self: Any,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor | None = None,
        *args: Any,
        sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        to_sync: list[torch.Tensor] = []

        def _fresh(t: torch.Tensor | None) -> torch.Tensor | None:
            if isinstance(t, torch.Tensor):
                c = t.contiguous().clone()
                to_sync.append(c)
                return c
            return t

        timestep = _fresh(timestep)  # type: ignore[assignment]
        audio_timestep = _fresh(audio_timestep)
        sigma = _fresh(sigma)
        audio_sigma = _fresh(audio_sigma)
        if to_sync:
            synchronize_tensors(to_sync, wait=False)

        res = orig_model_forward(
            self,
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            audio_encoder_hidden_states,
            timestep,
            audio_timestep,
            *args,
            sigma=sigma,
            audio_sigma=audio_sigma,
            **kwargs,
        )
        # Flush the post-block ulysses_pg SequenceParallelOutput lazy bucket before
        # LTXGuidanceExecutor enqueues its cfg_pg all_gather_into_tensor, preventing
        # DEFER_AND_FUSE from merging two incompatible subgroup device_assignments
        # ([0,1,4,5,6,7,2,3] vs [0,6,1,7,4,2,5,3] in device_assignment_pipeline.cc:160).
        out_tensors: list[torch.Tensor] = []
        if isinstance(res, tuple):
            out_tensors = [t for t in res if isinstance(t, torch.Tensor)]
        else:
            for attr in ("sample", "audio_sample"):
                t = getattr(res, attr, None)
                if isinstance(t, torch.Tensor):
                    out_tensors.append(t)
        if out_tensors:
            synchronize_tensors(out_tensors, wait=False)
        return res

    setattr(_fresh_ltx2_forward, _LTX2_COMPILE_PATCHED, True)
    model_cls.forward = _fresh_ltx2_forward


def apply_ltx2_compile_patches() -> None:
    """Register LTX-2 TPU setup_compile hook and layout/sync patches."""
    from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2
    from vllm_omni.diffusion.worker import diffusion_model_runner

    _patch_ltx2_async_guidance_step()
    _patch_ltx2_mxu_gated_attention()

    if getattr(pipeline_ltx2.LTX2Pipeline, "_tpu_setup_compile_registered", False):
        return

    def _ltx2_setup_compile(self: Any) -> None:
        _activate_ltx2_compile_runtime_hooks()
        if getattr(self, "transformer", None) is not None:
            _install_shared_block_compile(self.transformer)
        if getattr(self, "transformer_2", None) is not None:
            _install_shared_block_compile(self.transformer_2)

    pipeline_ltx2.LTX2Pipeline.setup_compile = _ltx2_setup_compile
    pipeline_ltx2.LTX2Pipeline._tpu_setup_compile_registered = True  # type: ignore[attr-defined]

    runner_cls = diffusion_model_runner.DiffusionModelRunner
    if not getattr(runner_cls._compile_transformer, "_tpu_backend_patched", False):
        from vllm_omni.diffusion.compile import regionally_compile

        def _tpu_compile_transformer(self: Any, attr_name: str) -> None:
            model = getattr(self.pipeline, attr_name, None)
            if model is None:
                return
            compile_kwargs = {"backend": "tpu", "dynamic": False}
            compile_granularity = self.od_config.diffusion_compile_granularity
            try:
                if compile_granularity == "full":
                    model.compile(**compile_kwargs)
                    compiled_model = model
                else:
                    compiled_model = regionally_compile(model, **compile_kwargs)
                setattr(self.pipeline, attr_name, compiled_model)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Model runner: %s TPU torch.compile setup for %s failed: %s.",
                    compile_granularity,
                    attr_name,
                    exc,
                )

        setattr(_tpu_compile_transformer, "_tpu_backend_patched", True)
        runner_cls._compile_transformer = _tpu_compile_transformer

    logger.info("Applied TPU patch: LTX-2 shared block torch.compile + stack setup.")
