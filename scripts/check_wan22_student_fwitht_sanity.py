#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-path sanity check for student_F_withT.

This script reuses `T2VDistillModel_rCM.student_F_withT` and compares:
1) primal output vs standalone student-F forward
2) tangent output vs central finite differences
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from einops import rearrange

from rcm.conditioner import TextCondition
from rcm.models.t2v_model_distill_rcm import T2VDistillModel_rCM
from rcm.networks.wan2pt2_t2v_jvp import WanModel_T2V_JVP
from rcm.utils.denoiser_scaling import RectifiedFlow_TrigFlowWrapper
from rcm.utils.model_utils import load_state_dict


def _resolve_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    out = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).detach()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel_l2 = (diff.pow(2).mean().sqrt() / a.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()
    return {"max_abs": max_abs, "mean_abs": mean_abs, "rel_l2": rel_l2}


class _StudentFWithTHarness:
    """Minimal object that exposes the fields required by student_F_withT."""

    student_F_withT = T2VDistillModel_rCM.student_F_withT

    def __init__(self, net: WanModel_T2V_JVP, device: str, dtype: torch.dtype, rf_t_scaling_factor: float):
        self.net = net
        self.tensor_kwargs = {"device": device, "dtype": dtype}
        self.scaling = RectifiedFlow_TrigFlowWrapper(sigma_data=1.0, t_scaling_factor=rf_t_scaling_factor)


def _student_f_forward(
    harness: _StudentFWithTHarness,
    xt_B_C_T_H_W: torch.Tensor,
    time_B_1: torch.Tensor,
    condition: TextCondition,
) -> torch.Tensor:
    time_B_1_T_1_1 = rearrange(time_B_1, "b t -> b 1 t 1 1")
    c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = harness.scaling(trigflow_t=time_B_1_T_1_1)

    x_input = (xt_B_C_T_H_W * c_in_B_1_T_1_1).to(**harness.tensor_kwargs)
    noise_input = c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(**harness.tensor_kwargs)
    net_output = harness.net(
        x_B_C_T_H_W=x_input,
        timesteps_B_T=noise_input,
        **condition.to_dict(),
    ).float()

    x0_pred = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output
    return (torch.cos(time_B_1_T_1_1) * xt_B_C_T_H_W - x0_pred) / torch.sin(time_B_1_T_1_1)


def _preset_to_model_cfg(preset: str) -> dict[str, int]:
    table = {
        "mini": dict(dim=128, ffn_dim=256, num_heads=8, num_layers=2),
        "small": dict(dim=256, ffn_dim=768, num_heads=8, num_layers=6),
        "medium": dict(dim=512, ffn_dim=1536, num_heads=8, num_layers=12),
    }
    if preset not in table:
        raise ValueError(f"Unsupported preset '{preset}'. Supported: {sorted(table.keys())}")
    return table[preset]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training-path sanity check for student_F_withT")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", type=str, default="small", choices=["mini", "small", "medium"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--in_dim", type=int, default=16)
    parser.add_argument("--out_dim", type=int, default=16)
    parser.add_argument("--freq_dim", type=int, default=256)
    parser.add_argument("--text_dim", type=int, default=4096)
    parser.add_argument("--text_len", type=int, default=512)
    parser.add_argument("--latent_t", type=int, default=3)
    parser.add_argument("--latent_h", type=int, default=8)
    parser.add_argument("--latent_w", type=int, default=8)
    parser.add_argument("--time_min", type=float, default=0.2, help="Lower bound for trigflow time.")
    parser.add_argument("--time_max", type=float, default=1.2, help="Upper bound for trigflow time.")
    parser.add_argument("--dt_scale", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--rf_t_scaling_factor", type=float, default=1000.0)
    parser.add_argument("--student_ckpt", type=str, default="")
    parser.add_argument("--ckpt_prefix", type=str, default="net.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--primal_max_abs_threshold", type=float, default=1e-4)
    parser.add_argument("--primal_rel_l2_threshold", type=float, default=1e-4)
    parser.add_argument("--tangent_max_abs_threshold", type=float, default=1e-3)
    parser.add_argument("--tangent_rel_l2_threshold", type=float, default=5e-3)
    parser.add_argument("--save_json", type=str, default="")
    return parser.parse_args()


@torch.no_grad()
def run_check(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device
    dtype = _resolve_dtype(args.dtype)
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        raise ValueError(f"dtype={args.dtype} is not supported on cpu for this check.")

    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    model_cfg = _preset_to_model_cfg(args.preset)
    net = WanModel_T2V_JVP(
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=args.text_len,
        in_dim=args.in_dim,
        dim=model_cfg["dim"],
        ffn_dim=model_cfg["ffn_dim"],
        freq_dim=args.freq_dim,
        text_dim=args.text_dim,
        out_dim=args.out_dim,
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        qk_norm=True,
        cross_attn_norm=True,
        naive_attn=True,
    ).to(device=device, dtype=dtype)
    net.eval()

    ckpt_info = {}
    if args.student_ckpt:
        state = load_state_dict(args.student_ckpt)
        state = _strip_prefix(state, args.ckpt_prefix)
        incompatible = net.load_state_dict(state, strict=False)
        ckpt_info = {
            "student_ckpt": args.student_ckpt,
            "missing_keys": len(incompatible.missing_keys),
            "unexpected_keys": len(incompatible.unexpected_keys),
        }

    harness = _StudentFWithTHarness(
        net=net,
        device=device,
        dtype=dtype,
        rf_t_scaling_factor=args.rf_t_scaling_factor,
    )

    xt = torch.randn(args.batch_size, args.in_dim, args.latent_t, args.latent_h, args.latent_w, device=device, dtype=dtype)
    crossattn_emb = torch.randn(args.batch_size, args.text_len, args.text_dim, device=device, dtype=dtype)
    condition = TextCondition(crossattn_emb=crossattn_emb)

    time = torch.rand(args.batch_size, 1, device=device, dtype=dtype) * (args.time_max - args.time_min) + args.time_min
    dxt = torch.randn_like(xt)
    dtime = torch.randn_like(time) * args.dt_scale

    F_primal = _student_f_forward(harness, xt_B_C_T_H_W=xt, time_B_1=time, condition=condition)
    F_withT_primal, F_withT_tangent = harness.student_F_withT(
        xt_B_C_T_H_W=(xt, dxt),
        time=(time, dtime),
        condition=condition,
    )
    primal_metrics = _metrics(F_primal, F_withT_primal)

    eps = args.eps
    F_pos = _student_f_forward(
        harness,
        xt_B_C_T_H_W=xt + eps * dxt,
        time_B_1=time + eps * dtime,
        condition=condition,
    )
    F_neg = _student_f_forward(
        harness,
        xt_B_C_T_H_W=xt - eps * dxt,
        time_B_1=time - eps * dtime,
        condition=condition,
    )
    F_fd = (F_pos - F_neg) / (2.0 * eps)
    tangent_metrics = _metrics(F_fd, F_withT_tangent)

    return {
        "config": {
            "device": device,
            "dtype": args.dtype,
            "preset": args.preset,
            "batch_size": args.batch_size,
            "latent_shape": [args.latent_t, args.latent_h, args.latent_w],
            "time_range": [args.time_min, args.time_max],
            "dt_scale": args.dt_scale,
            "eps": args.eps,
            "rf_t_scaling_factor": args.rf_t_scaling_factor,
            "model_cfg": model_cfg,
            "seed": args.seed,
            **ckpt_info,
        },
        "primal_consistency": primal_metrics,
        "tangent_consistency": tangent_metrics,
    }


def main() -> None:
    args = parse_args()
    result = run_check(args)
    primal = result["primal_consistency"]
    tangent = result["tangent_consistency"]

    print(
        "[primal] "
        f"max_abs={primal['max_abs']:.6e}, "
        f"mean_abs={primal['mean_abs']:.6e}, "
        f"rel_l2={primal['rel_l2']:.6e}"
    )
    print(
        "[tangent] "
        f"max_abs={tangent['max_abs']:.6e}, "
        f"mean_abs={tangent['mean_abs']:.6e}, "
        f"rel_l2={tangent['rel_l2']:.6e}"
    )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[save] wrote student_F_withT sanity report to {out_path}")

    if args.strict:
        if primal["max_abs"] > args.primal_max_abs_threshold or primal["rel_l2"] > args.primal_rel_l2_threshold:
            raise RuntimeError(
                "Primal consistency check failed: "
                f"max_abs={primal['max_abs']:.6e} (thr={args.primal_max_abs_threshold:.6e}), "
                f"rel_l2={primal['rel_l2']:.6e} (thr={args.primal_rel_l2_threshold:.6e})"
            )
        if tangent["max_abs"] > args.tangent_max_abs_threshold or tangent["rel_l2"] > args.tangent_rel_l2_threshold:
            raise RuntimeError(
                "Tangent consistency check failed: "
                f"max_abs={tangent['max_abs']:.6e} (thr={args.tangent_max_abs_threshold:.6e}), "
                f"rel_l2={tangent['rel_l2']:.6e} (thr={args.tangent_rel_l2_threshold:.6e})"
            )

    print("[done] student_F_withT sanity check completed")


if __name__ == "__main__":
    main()
