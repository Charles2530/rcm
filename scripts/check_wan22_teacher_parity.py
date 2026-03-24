#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity check for Wan2.2 T2V A14B teacher conversion.

This script compares random forward outputs between:
1) Diffusers Wan2.2 expert (`transformer` / `transformer_2`)
2) rCM `WanModel` loaded from converted checkpoints (`net.*`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rcm.networks.wan2pt2 import WanModel
from rcm.utils.model_utils import load_state_dict


def _import_wan_transformer_cls():
    try:
        from diffusers import WanTransformer3DModel

        return WanTransformer3DModel
    except Exception:
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

        return WanTransformer3DModel


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_wan22_t2v_config(transformer_config: Mapping[str, Any]) -> dict[str, Any]:
    num_heads = int(transformer_config["num_attention_heads"])
    head_dim = int(transformer_config["attention_head_dim"])
    qk_norm_raw = transformer_config.get("qk_norm", True)
    return dict(
        model_type="t2v",
        patch_size=tuple(transformer_config.get("patch_size", [1, 2, 2])),
        in_dim=int(transformer_config["in_channels"]),
        out_dim=int(transformer_config["out_channels"]),
        dim=num_heads * head_dim,
        ffn_dim=int(transformer_config["ffn_dim"]),
        freq_dim=int(transformer_config["freq_dim"]),
        text_dim=int(transformer_config["text_dim"]),
        num_heads=num_heads,
        num_layers=int(transformer_config["num_layers"]),
        eps=float(transformer_config.get("eps", 1e-6)),
        qk_norm=bool(qk_norm_raw),
        cross_attn_norm=bool(transformer_config.get("cross_attn_norm", True)),
        text_len=512,
    )


def _strip_prefix(state_dict: Mapping[str, Any], prefix: str = "net.") -> dict[str, Any]:
    out = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _unwrap_output(output: Any) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        output = output[0]
    if hasattr(output, "sample"):
        output = output.sample
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor output, got {type(output)}")
    return output


def _forward_diffusers(model, x, t, crossattn_emb) -> torch.Tensor:
    # Different diffusers versions expose slightly different parameter names.
    attempts = [
        dict(hidden_states=x, timestep=t, encoder_hidden_states=crossattn_emb, return_dict=False),
        dict(hidden_states=x, timestep=t, encoder_hidden_states=crossattn_emb),
        dict(sample=x, timestep=t, encoder_hidden_states=crossattn_emb, return_dict=False),
        dict(sample=x, timestep=t, encoder_hidden_states=crossattn_emb),
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return _unwrap_output(model(**kwargs))
        except TypeError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to call diffusers WanTransformer3DModel forward. Last error: {last_error}")


def _resolve_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


@torch.no_grad()
def _check_one_expert(
    model_root: Path,
    converted_ckpt: Path,
    subdir: str,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    prefix: str,
) -> dict[str, float]:
    transformer_dir = model_root / subdir
    cfg = _load_json(transformer_dir / "config.json")
    net_args = _parse_wan22_t2v_config(cfg)

    WanTransformer3DModel = _import_wan_transformer_cls()
    hf_model = WanTransformer3DModel.from_pretrained(str(transformer_dir), torch_dtype=dtype, local_files_only=True).to(device=device, dtype=dtype)
    hf_model.eval()

    rcm_model = WanModel(**net_args).to(device=device, dtype=dtype)
    rcm_model.eval()
    converted_state = _strip_prefix(load_state_dict(str(converted_ckpt)), prefix=prefix)
    incompatible = rcm_model.load_state_dict(converted_state, strict=False)
    print(
        f"[{subdir}] load converted checkpoint: "
        f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
    )

    x = torch.randn(batch_size, net_args["in_dim"], latent_t, latent_h, latent_w, device=device, dtype=dtype)
    t = torch.randint(low=0, high=1000, size=(batch_size,), device=device, dtype=torch.int32).to(dtype=dtype)
    crossattn_emb = torch.randn(batch_size, net_args["text_len"], net_args["text_dim"], device=device, dtype=dtype)

    out_hf = _forward_diffusers(hf_model, x, t, crossattn_emb).float()
    out_rcm = rcm_model(
        x_B_C_T_H_W=x,
        timesteps_B_T=t.view(batch_size, 1),
        crossattn_emb=crossattn_emb,
    ).float()

    if out_hf.shape != out_rcm.shape:
        raise RuntimeError(f"[{subdir}] shape mismatch: diffusers={tuple(out_hf.shape)} rcm={tuple(out_rcm.shape)}")

    diff = out_hf - out_rcm
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel_l2 = (diff.pow(2).mean().sqrt() / out_hf.pow(2).mean().sqrt().clamp(min=1e-8)).item()
    print(f"[{subdir}] max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rel_l2={rel_l2:.6e}")
    return {"max_abs": max_abs, "mean_abs": mean_abs, "rel_l2": rel_l2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Wan2.2 teacher conversion parity")
    parser.add_argument("--model_root", type=str, default="./model/Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument("--converted_root", type=str, default="./model/Wan2.2-T2V-A14B-Diffusers-rcm")
    parser.add_argument("--ckpt_transformer", type=str, default="Wan2.2-T2V-A14B-transformer-rcm.pth")
    parser.add_argument("--ckpt_transformer_2", type=str, default="Wan2.2-T2V-A14B-transformer_2-rcm.pth")
    parser.add_argument("--skip_transformer_2", action="store_true")
    parser.add_argument("--prefix", type=str, default="net.")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--latent_t", type=int, default=21)
    parser.add_argument("--latent_h", type=int, default=60)
    parser.add_argument("--latent_w", type=int, default=106)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="Fail when parity metrics exceed thresholds.")
    parser.add_argument("--max_abs_threshold", type=float, default=1e-3)
    parser.add_argument("--mean_abs_threshold", type=float, default=1e-4)
    parser.add_argument("--rel_l2_threshold", type=float, default=1e-3)
    return parser.parse_args()


def _check_thresholds(subdir: str, metrics: Mapping[str, float], args: argparse.Namespace) -> None:
    if metrics["max_abs"] > args.max_abs_threshold:
        raise RuntimeError(f"[{subdir}] max_abs {metrics['max_abs']:.6e} exceeds threshold {args.max_abs_threshold:.6e}")
    if metrics["mean_abs"] > args.mean_abs_threshold:
        raise RuntimeError(f"[{subdir}] mean_abs {metrics['mean_abs']:.6e} exceeds threshold {args.mean_abs_threshold:.6e}")
    if metrics["rel_l2"] > args.rel_l2_threshold:
        raise RuntimeError(f"[{subdir}] rel_l2 {metrics['rel_l2']:.6e} exceeds threshold {args.rel_l2_threshold:.6e}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model_root = Path(args.model_root)
    converted_root = Path(args.converted_root)
    dtype = _resolve_dtype(args.dtype)

    checks: Sequence[tuple[str, Path]] = [
        ("transformer", converted_root / args.ckpt_transformer),
    ]
    if not args.skip_transformer_2:
        checks = [*checks, ("transformer_2", converted_root / args.ckpt_transformer_2)]

    all_metrics = {}
    for subdir, ckpt_path in checks:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Converted checkpoint does not exist: {ckpt_path}")
        metrics = _check_one_expert(
            model_root=model_root,
            converted_ckpt=ckpt_path,
            subdir=subdir,
            device=args.device,
            dtype=dtype,
            batch_size=args.batch_size,
            latent_t=args.latent_t,
            latent_h=args.latent_h,
            latent_w=args.latent_w,
            prefix=args.prefix,
        )
        all_metrics[subdir] = metrics
        if args.strict:
            _check_thresholds(subdir, metrics, args)

    print("[done] parity check completed")
    for subdir, metrics in all_metrics.items():
        print(
            f"  - {subdir}: "
            f"max_abs={metrics['max_abs']:.6e}, "
            f"mean_abs={metrics['mean_abs']:.6e}, "
            f"rel_l2={metrics['rel_l2']:.6e}"
        )


if __name__ == "__main__":
    main()
