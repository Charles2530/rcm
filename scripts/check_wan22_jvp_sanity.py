#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanity check for Wan2.2 dense-student JVP path.

Runs two checks on a tiny T2V model:
1) Forward consistency: plain forward vs withT forward primal output
2) Tangent consistency: withT tangent vs central finite difference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rcm.networks.wan2pt2_t2v_jvp import WanModel_T2V_JVP


def _resolve_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).detach()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel_l2 = (diff.pow(2).mean().sqrt() / a.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()
    return {"max_abs": max_abs, "mean_abs": mean_abs, "rel_l2": rel_l2}


@torch.no_grad()
def run_check(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device
    dtype = _resolve_dtype(args.dtype)

    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        raise ValueError(f"dtype={args.dtype} is not supported on cpu for this check.")

    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    net = WanModel_T2V_JVP(
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=args.text_len,
        in_dim=args.in_dim,
        dim=args.dim,
        ffn_dim=args.ffn_dim,
        freq_dim=args.freq_dim,
        text_dim=args.text_dim,
        out_dim=args.out_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        qk_norm=True,
        cross_attn_norm=True,
        naive_attn=True,
    ).to(device=device, dtype=dtype)
    net.eval()

    x = torch.randn(
        args.batch_size,
        args.in_dim,
        args.latent_t,
        args.latent_h,
        args.latent_w,
        device=device,
        dtype=dtype,
    )
    t = torch.randint(0, 1000, (args.batch_size, 1), device=device, dtype=torch.int32).to(dtype=dtype)
    crossattn_emb = torch.randn(args.batch_size, args.text_len, args.text_dim, device=device, dtype=dtype)

    dx = torch.randn_like(x)
    dt = torch.randn_like(t)

    out_plain = net(x_B_C_T_H_W=x, timesteps_B_T=t, crossattn_emb=crossattn_emb)
    out_primal, out_tangent = net(
        x_B_C_T_H_W=(x, dx),
        timesteps_B_T=(t, dt),
        crossattn_emb=crossattn_emb,
        withT=True,
    )
    forward_metrics = _metrics(out_plain, out_primal)

    eps = args.eps
    out_pos = net(
        x_B_C_T_H_W=x + eps * dx,
        timesteps_B_T=t + eps * dt,
        crossattn_emb=crossattn_emb,
    )
    out_neg = net(
        x_B_C_T_H_W=x - eps * dx,
        timesteps_B_T=t - eps * dt,
        crossattn_emb=crossattn_emb,
    )
    out_fd = (out_pos - out_neg) / (2.0 * eps)
    tangent_metrics = _metrics(out_fd, out_tangent)

    result = {
        "config": {
            "device": device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "latent_shape": [args.latent_t, args.latent_h, args.latent_w],
            "model": {
                "in_dim": args.in_dim,
                "out_dim": args.out_dim,
                "dim": args.dim,
                "ffn_dim": args.ffn_dim,
                "freq_dim": args.freq_dim,
                "text_dim": args.text_dim,
                "text_len": args.text_len,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
            },
            "eps": eps,
            "seed": args.seed,
        },
        "forward_consistency": forward_metrics,
        "tangent_consistency": tangent_metrics,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check Wan2.2 T2V JVP path against finite differences")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--latent_t", type=int, default=3)
    parser.add_argument("--latent_h", type=int, default=8)
    parser.add_argument("--latent_w", type=int, default=8)
    parser.add_argument("--in_dim", type=int, default=4)
    parser.add_argument("--out_dim", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--ffn_dim", type=int, default=128)
    parser.add_argument("--freq_dim", type=int, default=32)
    parser.add_argument("--text_dim", type=int, default=64)
    parser.add_argument("--text_len", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--forward_max_abs_threshold", type=float, default=5e-4)
    parser.add_argument("--forward_rel_l2_threshold", type=float, default=5e-4)
    parser.add_argument("--tangent_max_abs_threshold", type=float, default=5e-3)
    parser.add_argument("--tangent_rel_l2_threshold", type=float, default=1e-2)
    parser.add_argument("--save_json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_check(args)

    fwd = result["forward_consistency"]
    tan = result["tangent_consistency"]
    print(
        "[forward] "
        f"max_abs={fwd['max_abs']:.6e}, mean_abs={fwd['mean_abs']:.6e}, rel_l2={fwd['rel_l2']:.6e}"
    )
    print(
        "[tangent] "
        f"max_abs={tan['max_abs']:.6e}, mean_abs={tan['mean_abs']:.6e}, rel_l2={tan['rel_l2']:.6e}"
    )

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[save] wrote JVP sanity report to {output_path}")

    if args.strict:
        if fwd["max_abs"] > args.forward_max_abs_threshold or fwd["rel_l2"] > args.forward_rel_l2_threshold:
            raise RuntimeError(
                "Forward consistency check failed: "
                f"max_abs={fwd['max_abs']:.6e} (thr={args.forward_max_abs_threshold:.6e}), "
                f"rel_l2={fwd['rel_l2']:.6e} (thr={args.forward_rel_l2_threshold:.6e})"
            )
        if tan["max_abs"] > args.tangent_max_abs_threshold or tan["rel_l2"] > args.tangent_rel_l2_threshold:
            raise RuntimeError(
                "Tangent consistency check failed: "
                f"max_abs={tan['max_abs']:.6e} (thr={args.tangent_max_abs_threshold:.6e}), "
                f"rel_l2={tan['rel_l2']:.6e} (thr={args.tangent_rel_l2_threshold:.6e})"
            )

    print("[done] JVP sanity check completed")


if __name__ == "__main__":
    main()
