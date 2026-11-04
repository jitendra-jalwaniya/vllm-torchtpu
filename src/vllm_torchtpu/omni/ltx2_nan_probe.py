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
"""Debug-only NaN probe for the vLLM-Omni LTX-2 pipeline on TPU.

Enabled with VLLM_TORCHTPU_LTX_NAN_PROBE=1. It writes one JSONL file per
rank to $LTX_PROBE_DIR (default /tmp/ltx_probe):

- kind=pass: per denoise step and modality, the non-finite count, abs-max
  and std of every guidance pass (cond / uncond / ptb / mod), the guided
  velocity, and the x0 rescale factor.
- kind=fwd: per transformer forward, the first non-finite sub-module per
  batch slot (= local guidance pass) and the abs-max of every block output.
- kind=tail: inputs and outputs of the video VAE decode, the audio VAE
  decode and the vocoder.

Every record of a forward is reduced on device and copied to the host once,
so the probe costs one sync per forward plus a few per guidance step.
"""

import json
import os
import re
import time
from dataclasses import replace
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PROBE_DIR = os.environ.get("LTX_PROBE_DIR", "/tmp/ltx_probe")
# Write per-block abs-max on these steps, on every 10th step and on any
# step with a non-finite value.
_FULL_STEPS = {0, 1}

_state: dict[str, Any] = {
    "step": -1,
    "slots": None,
    "modality_idx": 0,
    "buf": [],
    "names_set": False,
    "fh": None,
    "fwd": 0,
}

_BLOCK_RE = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)(?:\.(.*))?$")
_SUBMODULES = (
    "attn1",
    "audio_attn1",
    "attn2",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
    "ff",
    "audio_ff",
    "attn1.to_gate_logits",
    "audio_attn1.to_gate_logits",
    "attn2.to_gate_logits",
    "audio_attn2.to_gate_logits",
    "audio_to_video_attn.to_gate_logits",
    "video_to_audio_attn.to_gate_logits",
    "attn1.to_v",
    "audio_attn1.to_v",
)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _write(record: dict[str, Any]) -> None:
    if _state["fh"] is None:
        os.makedirs(_PROBE_DIR, exist_ok=True)
        path = os.path.join(_PROBE_DIR, f"probe_rank{_rank()}.jsonl")
        _state["fh"] = open(path, "a", buffering=1)
    record["rank"] = _rank()
    record["t"] = round(time.time(), 3)
    _state["fh"].write(json.dumps(record) + "\n")


def _first_tensor(out: Any) -> torch.Tensor | None:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    return None


def _slot_stats(x: torch.Tensor) -> torch.Tensor:
    """(2, B) float32: non-finite count and abs-max per batch slot."""
    flat = x.detach().reshape(x.shape[0], -1)
    bad = (~torch.isfinite(flat)).sum(dim=1).float()
    amax = flat.float().abs().amax(dim=1)
    return torch.stack([bad, amax])


def _scalar_stats(x: torch.Tensor) -> torch.Tensor:
    """(3,) float32: non-finite count, abs-max and std over the tensor."""
    flat = x.detach().reshape(-1).float()
    return torch.stack(
        [(~torch.isfinite(flat)).sum().float(), flat.abs().amax(), flat.std()]
    )


def _to_list(values: torch.Tensor) -> list[Any]:
    return [round(v, 5) if v == v else "nan" for v in values.cpu().tolist()]


def _install_guidance_probe() -> None:
    from vllm_omni.diffusion.models.ltx2 import ltx2_guidance as g

    executor_cls = g.LTXGuidanceExecutor
    orig_predict_noise = executor_cls.predict_noise
    orig_guide = executor_cls._guide_modality
    orig_combine = g.combine_guided_x0

    def predict_noise(self, pipeline, plan, index, timestep, state, forward_ctx, denoise_ctx, *args, **kwargs):
        _state["step"] = int(index)
        _state["modality_idx"] = 0
        if forward_ctx.guidance_parallel_ready:
            world = g.get_guidance_parallel_world_size()
            rank = g.get_guidance_parallel_rank()
            assignments = executor_cls._parallel_assignments(len(plan.passes), world)
            local = assignments[rank]
            width = max(len(a) for a in assignments)
            names = [plan.passes[i].name for i in local]
            names += ["pad(cond)"] * (width - len(local))
        else:
            names = list(plan.names)
        _state["slots"] = names
        return orig_predict_noise(
            self, pipeline, plan, index, timestep, state, forward_ctx, denoise_ctx, *args, **kwargs
        )

    def guide_modality(sample, splits, sigma, guidance, **kwargs):
        modality = "video" if _state["modality_idx"] == 0 else "audio"
        _state["modality_idx"] += 1
        names = list(splits)
        stats = torch.stack([_scalar_stats(splits[n]) for n in names] + [_scalar_stats(sample)])
        out = orig_guide(sample, splits, sigma, guidance, **kwargs)
        out_stats = _scalar_stats(out)
        host = stats.cpu().tolist()
        record = {
            "kind": "pass",
            "step": _state["step"],
            "modality": modality,
            "guidance": {
                "cfg": guidance.cfg_scale,
                "stg": guidance.stg_scale,
                "mod": guidance.modality_scale,
                "rescale": guidance.rescale_scale,
            },
            "passes": {
                n: {"bad": int(h[0]), "amax": h[1], "std": h[2]} for n, h in zip(names, host[:-1])
            },
            "sample": {"bad": int(host[-1][0]), "amax": host[-1][1], "std": host[-1][2]},
            "guided": dict(zip(("bad", "amax", "std"), _to_list(out_stats))),
            "rescale": _state.pop("last_rescale", None),
        }
        _write(record)
        return out

    def combine_guided_x0(*, cond, uncond_text, uncond_perturbed, uncond_modality, guidance, rescale_token_count=None):
        if guidance.rescale_scale != 0.0:
            raw = orig_combine(
                cond=cond,
                uncond_text=uncond_text,
                uncond_perturbed=uncond_perturbed,
                uncond_modality=uncond_modality,
                guidance=replace(guidance, rescale_scale=0.0),
                rescale_token_count=rescale_token_count,
            ).float()
            c = cond.float()
            if rescale_token_count is not None:
                c = c[:, :rescale_token_count]
                raw = raw[:, :rescale_token_count]
            stats = torch.stack(
                [
                    c.std(),
                    raw.std(),
                    (~torch.isfinite(raw)).sum().float(),
                    raw.abs().amax(),
                ]
            )
            _state["last_rescale"] = dict(
                zip(("cond_std", "pred_std", "pred_bad", "pred_amax"), _to_list(stats))
            )
        return orig_combine(
            cond=cond,
            uncond_text=uncond_text,
            uncond_perturbed=uncond_perturbed,
            uncond_modality=uncond_modality,
            guidance=guidance,
            rescale_token_count=rescale_token_count,
        )

    executor_cls.predict_noise = predict_noise
    executor_cls._guide_modality = staticmethod(guide_modality)
    g.combine_guided_x0 = combine_guided_x0


def _make_hook(name: str):
    def hook(module, args, output):
        if output is None:
            return
        if name.endswith(".out"):
            hs, audio_hs = output
            _state["buf"].append((name + ".v", _slot_stats(hs)))
            _state["buf"].append((name + ".a", _slot_stats(audio_hs)))
            return
        x = _first_tensor(output)
        if x is not None and x.ndim >= 2:
            _state["buf"].append((name, _slot_stats(x)))

    return hook


def _register_block_hooks(model: torch.nn.Module) -> None:
    # LTX_PROBE_SUBMODULES limits the hooks (comma list of _SUBMODULES
    # entries, plus "out" for block outputs), to bisect which hook hides
    # the bug.
    only = os.environ.get("LTX_PROBE_SUBMODULES")
    allowed = set(only.split(",")) if only else set(_SUBMODULES) | {"out"}
    count = 0
    for full_name, module in model.named_modules():
        match = _BLOCK_RE.search(full_name)
        if match is None:
            continue
        block, sub = int(match.group(1)), match.group(2)
        if sub is None and "out" in allowed:
            module.register_forward_hook(_make_hook(f"b{block}.out"))
        elif sub is not None and sub in _SUBMODULES and sub in allowed:
            module.register_forward_hook(_make_hook(f"b{block}.{sub}"))
        else:
            continue
        count += 1
    logger.info("LTX NaN probe: registered %d block hooks.", count)


def _flush_forward(inputs: dict[str, torch.Tensor], outputs: Any) -> None:
    buf = _state["buf"]
    _state["buf"] = []
    if not buf:
        return
    if os.environ.get("LTX_PROBE_NOSYNC") == "1":
        # Keep the hooks' device reductions but never read them back.
        _state["fwd"] += 1
        return
    names = [n for n, _ in buf]
    batch = buf[0][1].shape[1]
    stacked = torch.stack([t for _, t in buf if t.shape[1] == batch])
    names = [n for n, t in buf if t.shape[1] == batch]
    in_stats = {k: _to_list(_scalar_stats(v)) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    host = stacked.cpu()  # (N, 2, B)
    step = _state["step"]
    slots = _state["slots"] or [f"slot{i}" for i in range(batch)]
    if len(slots) != batch:
        slots = [f"slot{i}" for i in range(batch)]
    first_bad: dict[str, Any] = {}
    any_bad = False
    for b, slot in enumerate(slots):
        idx = torch.nonzero(host[:, 0, b] > 0)
        if idx.numel():
            i = int(idx[0])
            first_bad[slot] = {"op": names[i], "bad": int(host[i, 0, b]), "prev_op": names[i - 1] if i else None}
            any_bad = True
        else:
            first_bad[slot] = None
    record: dict[str, Any] = {
        "kind": "fwd",
        "fwd": _state["fwd"],
        "step": step,
        "slots": slots,
        "first_nonfinite": first_bad,
        "inputs": in_stats,
    }
    if any_bad or step in _FULL_STEPS or step % 10 == 0:
        per_op: dict[str, list[Any]] = {}
        for i, name in enumerate(names):
            per_op[name] = [round(float(v), 3) if v == v else "nan" for v in host[i, 1].tolist()]
        record["amax"] = per_op
    _write(record)
    _state["fwd"] += 1


def _install_transformer_probe() -> None:
    from vllm_omni.diffusion.models.ltx2 import ltx2_transformer as t

    model_cls = t.LTX2VideoTransformer3DModel
    orig_forward = model_cls.forward

    def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states, audio_encoder_hidden_states, *args, **kwargs):
        if not getattr(self, "_ltx_probe_hooked", False):
            _register_block_hooks(self)
            self._ltx_probe_hooked = True
        _state["buf"] = []
        inputs = {
            "hidden_states": hidden_states,
            "audio_hidden_states": audio_hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "audio_encoder_hidden_states": audio_encoder_hidden_states,
        }
        out = orig_forward(
            self, hidden_states, audio_hidden_states, encoder_hidden_states, audio_encoder_hidden_states, *args, **kwargs
        )
        _flush_forward(inputs, out)
        return out

    model_cls.forward = forward


def _wrap_tail(cls: Any, method: str, label: str) -> None:
    if cls is None or not hasattr(cls, method):
        return
    orig = getattr(cls, method)
    if getattr(orig, "_ltx_probe", False):
        return

    def wrapped(self, *args, **kwargs):
        x = args[0] if args else next(iter(kwargs.values()), None)
        out = orig(self, *args, **kwargs)
        y = _first_tensor(out)
        if y is None and hasattr(out, "sample"):
            y = out.sample
        record = {"kind": "tail", "op": label, "step": _state["step"]}
        if isinstance(x, torch.Tensor):
            record["in"] = dict(zip(("bad", "amax", "std"), _to_list(_scalar_stats(x))))
            record["in_shape"] = list(x.shape)
        if isinstance(y, torch.Tensor):
            record["out"] = dict(zip(("bad", "amax", "std"), _to_list(_scalar_stats(y))))
            record["out_shape"] = list(y.shape)
        _write(record)
        return out

    wrapped._ltx_probe = True
    setattr(cls, method, wrapped)


def _install_tail_probe() -> None:
    try:
        from diffusers.pipelines.ltx2 import vocoder as voc

        _wrap_tail(getattr(voc, "LTX2Vocoder", None), "forward", "vocoder")
        _wrap_tail(getattr(voc, "LTX2VocoderWithBWE", None), "forward", "vocoder_bwe")
    except ImportError:
        pass
    try:
        from diffusers.models.autoencoders import autoencoder_kl_ltx2 as vae

        _wrap_tail(getattr(vae, "AutoencoderKLLTX2Video", None), "decode", "video_vae_decode")
    except ImportError:
        pass
    try:
        from diffusers.models.autoencoders import autoencoder_kl_ltx2_audio as avae

        _wrap_tail(getattr(avae, "AutoencoderKLLTX2Audio", None), "decode", "audio_vae_decode")
    except ImportError:
        pass


def install() -> None:
    # LTX_PROBE_LEVELS picks the probe parts, so the host syncs each part
    # adds can be bisected: any subset of "pass,fwd,tail".
    levels = set(os.environ.get("LTX_PROBE_LEVELS", "pass,fwd,tail").split(","))
    if "pass" in levels:
        _install_guidance_probe()
    if "fwd" in levels:
        _install_transformer_probe()
    if "tail" in levels:
        _install_tail_probe()
    logger.info("Applied debug patch: LTX NaN probe %s, writing to %s.", sorted(levels), _PROBE_DIR)
