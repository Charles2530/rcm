#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-level smoke test for dual-teacher routing in `denoise(net_type="teacher")`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from rcm.conditioner import TextCondition
from rcm.models.t2v_model_distill_rcm import T2VDistillModel_rCM
from rcm.networks.wan2pt2 import WanModel
from rcm.utils.denoiser_scaling import RectifiedFlow_TrigFlowWrapper
from rcm.utils.model_utils import load_state_dict
from rcm.utils.timestep_utils import rf_to_trig_time


def _resolve_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


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


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).detach()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel_l2 = (diff.pow(2).mean().sqrt() / a.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()
    return {"max_abs": max_abs, "mean_abs": mean_abs, "rel_l2": rel_l2}


def _build_route_noise_labels(
    *,
    batch_size: int,
    threshold: float,
    t_scaling_factor: float,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    # Seed two anchor samples near the boundary and fill the rest uniformly.
    labels: list[float] = []
    if batch_size >= 1:
        labels.append(max(0.0, threshold - 5.0))
    if batch_size >= 2:
        labels.append(min(t_scaling_factor, threshold + 5.0))

    while len(labels) < batch_size:
        labels.append(float(torch.rand((), device="cpu").item() * t_scaling_factor))

    tensor = torch.tensor(labels[:batch_size], device=device, dtype=dtype).view(batch_size, 1)
    return tensor


class _TeacherPipelineHarness:
    _slice_condition = T2VDistillModel_rCM._slice_condition
    _teacher_2_mask_from_noise_labels = T2VDistillModel_rCM._teacher_2_mask_from_noise_labels
    _forward_net = T2VDistillModel_rCM._forward_net
    _forward_teacher_dual_by_noise_labels = T2VDistillModel_rCM._forward_teacher_dual_by_noise_labels
    _forward_teacher_dual = T2VDistillModel_rCM._forward_teacher_dual
    denoise = T2VDistillModel_rCM.denoise

    def __init__(
        self,
        *,
        net_teacher: torch.nn.Module,
        net_teacher_2: torch.nn.Module,
        teacher_boundary_ratio: float,
        rectified_flow_t_scaling_factor: float,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.net_teacher = net_teacher
        self.net_teacher_2 = net_teacher_2
        self.tensor_kwargs = {"device": device, "dtype": dtype}
        self.scaling = RectifiedFlow_TrigFlowWrapper(
            sigma_data=1.0,
            t_scaling_factor=rectified_flow_t_scaling_factor,
        )
        self.config = SimpleNamespace(
            teacher_boundary_ratio=teacher_boundary_ratio,
            rectified_flow_t_scaling_factor=rectified_flow_t_scaling_factor,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline smoke test for Wan2.2 dual-teacher routing")
    parser.add_argument("--model_root", type=str, default="./model/Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument("--converted_root", type=str, default="./model/Wan2.2-T2V-A14B-Diffusers-rcm")
    parser.add_argument("--ckpt_transformer", type=str, default="Wan2.2-T2V-A14B-transformer-rcm.pth")
    parser.add_argument("--ckpt_transformer_2", type=str, default="Wan2.2-T2V-A14B-transformer_2-rcm.pth")
    parser.add_argument("--prefix", type=str, default="net.")
    parser.add_argument("--teacher_boundary_ratio", type=float, default=0.875)
    parser.add_argument("--rectified_flow_t_scaling_factor", type=float, default=1000.0)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--latent_t", type=int, default=21)
    parser.add_argument("--latent_h", type=int, default=60)
    parser.add_argument("--latent_w", type=int, default=106)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--boundary_deltas", type=str, default="1,5,10")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--route_mismatch_max_abs_threshold", type=float, default=1e-6)
    parser.add_argument("--route_mismatch_rel_l2_threshold", type=float, default=1e-6)
    parser.add_argument("--boundary_jump_rel_l2_warn_threshold", type=float, default=0.5)
    parser.add_argument("--save_json", type=str, default="")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    dtype = _resolve_dtype(args.dtype)
    if args.device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        raise ValueError(f"dtype={args.dtype} is not supported on cpu for this check.")

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    cfg_path = Path(args.model_root) / "transformer" / "config.json"
    cfg = _load_json(cfg_path)
    net_args = _parse_wan22_t2v_config(cfg)

    net_teacher = WanModel(**net_args).to(device=args.device, dtype=dtype).eval()
    net_teacher_2 = WanModel(**net_args).to(device=args.device, dtype=dtype).eval()

    ckpt_1 = _strip_prefix(load_state_dict(str(Path(args.converted_root) / args.ckpt_transformer)), args.prefix)
    ckpt_2 = _strip_prefix(load_state_dict(str(Path(args.converted_root) / args.ckpt_transformer_2)), args.prefix)
    inc_1 = net_teacher.load_state_dict(ckpt_1, strict=False)
    inc_2 = net_teacher_2.load_state_dict(ckpt_2, strict=False)
    print(
        f"[load] teacher_1 missing={len(inc_1.missing_keys)} unexpected={len(inc_1.unexpected_keys)} | "
        f"teacher_2 missing={len(inc_2.missing_keys)} unexpected={len(inc_2.unexpected_keys)}"
    )

    harness = _TeacherPipelineHarness(
        net_teacher=net_teacher,
        net_teacher_2=net_teacher_2,
        teacher_boundary_ratio=args.teacher_boundary_ratio,
        rectified_flow_t_scaling_factor=args.rectified_flow_t_scaling_factor,
        device=args.device,
        dtype=dtype,
    )

    x = torch.randn(args.batch_size, net_args["in_dim"], args.latent_t, args.latent_h, args.latent_w, device=args.device, dtype=dtype)
    crossattn_emb = torch.randn(args.batch_size, net_args["text_len"], net_args["text_dim"], device=args.device, dtype=dtype)
    condition = TextCondition(crossattn_emb=crossattn_emb)

    threshold = args.teacher_boundary_ratio * args.rectified_flow_t_scaling_factor
    noise_labels = _build_route_noise_labels(
        batch_size=args.batch_size,
        threshold=threshold,
        t_scaling_factor=args.rectified_flow_t_scaling_factor,
        device=args.device,
        dtype=dtype,
    )

    dual_out = harness._forward_teacher_dual_by_noise_labels(
        x_B_C_T_H_W=x,
        timesteps_B_T=noise_labels,
        noise_labels_B_T=noise_labels,
        condition=condition,
    )
    out_1 = harness._forward_net(net_teacher, x, noise_labels, condition)
    out_2 = harness._forward_net(net_teacher_2, x, noise_labels, condition)
    mask_2 = harness._teacher_2_mask_from_noise_labels(noise_labels)
    expected = torch.where(mask_2.view(-1, 1, 1, 1, 1), out_2, out_1)
    route_mismatch = _metrics(expected, dual_out)
    print(
        "[route] teacher_2_hit_ratio="
        f"{mask_2.float().mean().item():.3f}, "
        f"mismatch_max_abs={route_mismatch['max_abs']:.6e}, mismatch_rel_l2={route_mismatch['rel_l2']:.6e}"
    )

    jump_rows = []
    for delta in [float(x.strip()) for x in args.boundary_deltas.split(",") if x.strip()]:
        low_label = max(0.0, threshold - delta)
        high_label = min(args.rectified_flow_t_scaling_factor, threshold + delta)

        rf_low = torch.full((args.batch_size, 1), low_label / args.rectified_flow_t_scaling_factor, device=args.device, dtype=torch.float64)
        rf_high = torch.full((args.batch_size, 1), high_label / args.rectified_flow_t_scaling_factor, device=args.device, dtype=torch.float64)
        trig_low = rf_to_trig_time(rf_low).to(dtype=dtype)
        trig_high = rf_to_trig_time(rf_high).to(dtype=dtype)

        F_low = harness.denoise(x, trig_low, condition, net_type="teacher").F
        F_high = harness.denoise(x, trig_high, condition, net_type="teacher").F
        jump = _metrics(F_low, F_high)
        jump_rows.append(
            {
                "delta_noise_label": delta,
                "low_noise_label": low_label,
                "high_noise_label": high_label,
                **jump,
            }
        )
        print(
            f"[boundary] delta={delta:.3f}, low={low_label:.3f}, high={high_label:.3f}, "
            f"jump_mean_abs={jump['mean_abs']:.6e}, jump_rel_l2={jump['rel_l2']:.6e}"
        )

    report = {
        "config": {
            "teacher_boundary_ratio": args.teacher_boundary_ratio,
            "rectified_flow_t_scaling_factor": args.rectified_flow_t_scaling_factor,
            "threshold_noise_label": threshold,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "latent_shape": [args.latent_t, args.latent_h, args.latent_w],
            "seed": args.seed,
            "boundary_deltas": args.boundary_deltas,
        },
        "route_check": {
            "teacher_2_hit_ratio": float(mask_2.float().mean().item()),
            "mismatch_metrics": route_mismatch,
        },
        "boundary_jump_check": jump_rows,
    }

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[save] wrote teacher pipeline smoke report to {output_path}")

    if args.strict:
        if route_mismatch["max_abs"] > args.route_mismatch_max_abs_threshold or route_mismatch["rel_l2"] > args.route_mismatch_rel_l2_threshold:
            raise RuntimeError(
                "Routing mismatch check failed: "
                f"max_abs={route_mismatch['max_abs']:.6e} (thr={args.route_mismatch_max_abs_threshold:.6e}), "
                f"rel_l2={route_mismatch['rel_l2']:.6e} (thr={args.route_mismatch_rel_l2_threshold:.6e})"
            )
        max_jump_rel_l2 = max(row["rel_l2"] for row in jump_rows)
        if max_jump_rel_l2 > args.boundary_jump_rel_l2_warn_threshold:
            raise RuntimeError(
                "Boundary jump check exceeded threshold: "
                f"max_rel_l2={max_jump_rel_l2:.6e} (thr={args.boundary_jump_rel_l2_warn_threshold:.6e})"
            )

    print("[done] teacher pipeline smoke check completed")


if __name__ == "__main__":
    main()
