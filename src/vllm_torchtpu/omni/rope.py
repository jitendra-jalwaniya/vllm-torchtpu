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
"""TPU-friendly rotary embeddings for vLLM-Omni Wan and LTX-2 models.

``RotaryEmbeddingWan.forward_native`` rotates (even, odd) pairs through
``x.unflatten(-1, (-1, 2)).unbind(-1)`` and ``torch.stack(..., dim=-1)``.
On TPU the size-2 minor dimension forces sequence-minor relayouts plus
strided copies around every q/k rotation (the XLA "data formatting" and
copy ops around each attention layer in a device profile).

The same rotation is written here on the natural ``(..., head_dim)``
layout::

    out = x * cos2 + (x @ P) * sin2

where ``cos2``/``sin2`` repeat each half-dim frequency for its pair and
``P`` is the signed pair-swap permutation (``(x @ P)[2i] = -x[2i+1]``,
``(x @ P)[2i+1] = x[2i]``). The permutation runs on the otherwise idle
MXU and is exact (one +-1 term per output), so the result matches the
native formula term for term. On a v7x Wan2.2-A14B block at 480p (32760
tokens) this cuts ~7 ms per transformer layer (80.8 -> 73.9 ms).
"""

import functools
import os
from functools import cache as run_once

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.diffusion_attention.qk_norm_rope import qk_norm_rope

_USE_MXU_SPLIT_ROPE = os.environ.get("VLLM_TORCHTPU_LTX2_MXU_ROPE", "1") != "0"


@functools.cache
def _pair_swap_matrix(
    dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    p = torch.zeros(dim, dim, dtype=torch.float32)
    even = torch.arange(0, dim, 2)
    p[even + 1, even] = -1.0
    p[even, even + 1] = 1.0
    return p.to(device=device, dtype=dtype)


def _repeat_pairs(t: torch.Tensor) -> torch.Tensor:
    """``(..., D/2) -> (..., D)`` repeating each entry for its (even, odd) pair."""
    return t.unsqueeze(-1).expand(*t.shape, 2).flatten(-2)


def apply_rotary_pair_swap(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Interleaved RoPE of ``x`` (..., D) with half-dim ``cos``/``sin`` (..., D/2).

    In ``torch.bfloat16`` (the dtype Wan and LingBot-World pass after
    ``to(x.dtype)``), ``x @ P`` is bit-exact because each column of ``P``
    has a single ``+-1`` entry. In ``torch.float32``, TPU matmuls default
    to bf16 input rounding unless high precision is configured.
    """
    swapped = torch.matmul(x, _pair_swap_matrix(x.shape[-1], x.dtype, x.device))
    return (x * _repeat_pairs(cos) + swapped * _repeat_pairs(sin)).to(x.dtype)


def rotary_embedding_wan_forward(
    self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Drop-in ``RotaryEmbeddingWan.forward`` for bf16 interleaved half-dim tables."""
    if not (self.interleaved and self.half_head_dim) or x.dtype != torch.bfloat16:
        return self._forward_method(x, cos, sin)
    # vLLM-Omni 0.32 (vllm-omni#8112) lets callers such as LingBot-World pass
    # (tokens, D/2) tables; expand them to (1, S, 1, D/2) as forward_native does.
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    return apply_rotary_pair_swap(x, cos, sin)


@functools.cache
def _first_half_lanes(width: int, head_dim: int, device: torch.device) -> torch.Tensor:
    return torch.arange(width, device=device) % head_dim < head_dim // 2


def _build_half_swap_perm(
    block_dim: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build block-diagonal ``(block_dim, block_dim)`` half-swap permutation on ``device``."""
    half = head_dim // 2
    idx = torch.arange(block_dim, device=device)
    in_head = idx % head_dim
    swapped_idx = (idx - in_head) + ((in_head + half) % head_dim)
    return (idx.unsqueeze(1) == swapped_idx.unsqueeze(0)).to(dtype=dtype)


@functools.cache
def _half_swap_matrix(
    block_dim: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return _build_half_swap_perm(block_dim, head_dim, dtype, device)


def split_rope_natural_tables(
    cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """LTX-2 split-RoPE tables ``(B, H, T, D/2) -> (B, 1, T, H * D)``.

    Lays the tables out like q/k (head-major channels): entries ``j`` and
    ``j + D/2`` of a head share half-dim frequency ``j``, and the sine table
    carries the sign of each half (``-sin`` then ``sin``). T stays at dim 2,
    where LTX-2's sequence-parallel plan splits the tables.
    """
    b, heads, tokens, half = cos.shape

    def natural(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return (
            torch.cat([first, second], dim=-1)
            .transpose(1, 2)
            .reshape(b, 1, tokens, heads * 2 * half)
        )

    return natural(cos, cos), natural(-sin, sin)


def apply_split_rotary_half_swap(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """Split RoPE of ``x`` (B, T, H * D) with ``split_rope_natural_tables`` tables.

    For ``torch.bfloat16`` inputs, swaps each head's two halves via a 128-lane
    aligned ``(128, 128)`` block-diagonal MXU permutation matmul (1 head per
    block when ``D == 128``, 2 heads packed per block when ``D == 64``), avoiding
    unaligned ``x.roll(±D/2, dims=-1)`` slices and ``ConcatBitcast`` custom calls
    while remaining bit-exact (one ``1.0`` entry per column).
    """
    width = x.shape[-1]
    if _USE_MXU_SPLIT_ROPE and x.dtype == torch.bfloat16:
        block_dim = (
            128
            if (head_dim < 128 and 128 % head_dim == 0 and width % 128 == 0)
            else head_dim
        )
        perm = (
            _build_half_swap_perm(block_dim, head_dim, x.dtype, x.device)
            if torch.compiler.is_compiling()
            else _half_swap_matrix(block_dim, head_dim, x.dtype, x.device)
        )
        swapped = torch.matmul(x.reshape(-1, block_dim), perm).reshape(x.shape)
        out = torch.addcmul(x * cos.squeeze(1), swapped, sin.squeeze(1))
        return out.to(x.dtype)

    half = head_dim // 2
    lanes = (
        torch.arange(width, device=x.device) % head_dim < half
        if torch.compiler.is_compiling()
        else _first_half_lanes(width, head_dim, x.device)
    )
    swapped = torch.where(
        lanes,
        x.roll(-half, dims=-1),
        x.roll(half, dims=-1),
    )
    out = x * cos.squeeze(1)
    out.addcmul_(swapped, sin.squeeze(1))
    return out.to(x.dtype)


@run_once
def _qk_norm_rope_op():
    """Builds the torch custom op wrapping the fused Pallas kernel (once)."""

    def wan_qk_norm_rope(
        qkv: jax.Array,
        wq: jax.Array,
        wk: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        eps: float,
        head_dim: int,
    ) -> tuple[jax.Array, jax.Array]:
        return qk_norm_rope(qkv, wq, wk, cos, sin, eps=eps, head_dim=head_dim)

    op = pallas.jax_op("pallas::wan_qk_norm_rope", wan_qk_norm_rope)

    def fake(qkv, wq, wk, cos, sin, eps, head_dim):
        shape = (qkv.shape[0], qkv.shape[1] // 3)
        return qkv.new_empty(shape), qkv.new_empty(shape)

    op.register_fake(fake)
    return op


def can_fuse_qk_norm_rope(
    attn, hidden_states: torch.Tensor, rotary_emb: tuple | None
) -> bool:
    """Whether a WanSelfAttention call fits the fused q/k norm + RoPE kernel."""
    # kernel supports bf16 MHA with per-row RMSNorm, no gate compression,
    # head_dim % 128 == 0, half-dim RoPE tables.
    from vllm_omni.diffusion.layers.norm import RMSNorm

    return (
        hidden_states.dtype == torch.bfloat16
        and rotary_emb is not None
        and attn.to_gate_compress is None
        and attn.num_kv_heads == attn.num_heads
        and attn.head_dim % 128 == 0
        and isinstance(attn.norm_q, RMSNorm)
        and isinstance(attn.norm_k, RMSNorm)
    )


def wan_self_attention_fused(
    attn, hidden_states: torch.Tensor, rotary_emb: tuple, attn_metadata=None
) -> torch.Tensor:
    """``WanSelfAttention.forward`` with the fused q/k norm + RoPE kernel.

    Same computation as vLLM-Omni's forward (QKV projection, q/k RMSNorm over
    all heads, interleaved half-dim RoPE, attention, output projection).
    """
    qkv, _ = attn.to_qkv(hidden_states)
    batch, seq, width = qkv.shape
    cos, sin = rotary_emb  # (1 or batch, seq, 1, head_dim / 2)

    def table(t: torch.Tensor) -> torch.Tensor:
        t = _repeat_pairs(t.reshape(t.shape[0], seq, -1))
        return t.expand(batch, seq, attn.head_dim).reshape(batch * seq, -1)

    query, key = _qk_norm_rope_op()(
        qkv.reshape(batch * seq, width),
        attn.norm_q.weight,
        attn.norm_k.weight,
        table(cos),
        table(sin),
        attn.norm_q.variance_epsilon,
        attn.head_dim,
    )
    heads = (attn.num_heads, attn.head_dim)
    query = query.reshape(batch, seq, *heads)
    key = key.reshape(batch, seq, *heads)
    value = qkv[..., 2 * attn.num_heads * attn.head_dim :].unflatten(2, heads)
    out = attn.attn(query, key, value, attn_metadata)
    out = out.flatten(2, 3).type_as(query)
    return attn.dropout(attn.to_out(out))
