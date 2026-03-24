#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-state preflight check for Wan2.2 A14B prototype distillation.

This script instantiates the real training model (`T2VDistillModel_rCM`) via
the same config entrypoint as training, then runs:
1) student init consistency check on the instantiated model object
2) `student_F_withT` primal/tangent consistency check on the instantiated model
3) teacher pipeline smoke in `denoise(net_type="teacher")` near route boundary
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import torch
from einops import rearrange

from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import get_config_module, override
from rcm.conditioner import TextCondition
from rcm.models.t2v_model_distill_rcm import T2VDistillModel_rCM
from rcm.utils.timestep_utils import rf_to_trig_time


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).detach()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel_l2 = (diff.pow(2).mean().sqrt() / a.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()
    return {"max_abs": max_abs, "mean_abs": mean_abs, "rel_l2": rel_l2}


def _resolve_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def _parse_csv_floats(csv: str) -> list[float]:
    vals = [s.strip() for s in csv.split(",") if s.strip()]
    return [float(v) for v in vals]


def _load_model_from_config(args: argparse.Namespace) -> T2VDistillModel_rCM:
    cfg_module_name = get_config_module(args.config)
    cfg_module = importlib.import_module(cfg_module_name)
    config = cfg_module.make_config()

    opts = [
        f"experiment={args.experiment}",
        "model.config.fsdp_shard_size=1",
        "model_parallel.context_parallel_size=1",
        f"model.config.precision={args.precision}",
        f"model.config.teacher_ckpt={args.teacher_ckpt}",
        f"model.config.teacher_ckpt_2={args.teacher_ckpt_2}",
        f"model.config.teacher_init_strategy={args.teacher_init_strategy}",
        f"model.config.teacher_init_low_noise_weight={args.teacher_init_low_noise_weight}",
        f"model.config.teacher_init_module_aware={str(args.teacher_init_module_aware).lower()}",
        f"model.config.teacher_init_low_noise_weight_embed={args.teacher_init_low_noise_weight_embed}",
        f"model.config.teacher_init_low_noise_weight_early={args.teacher_init_low_noise_weight_early}",
        f"model.config.teacher_init_low_noise_weight_late={args.teacher_init_low_noise_weight_late}",
        f"model.config.teacher_init_low_noise_weight_head={args.teacher_init_low_noise_weight_head}",
        f"model.config.teacher_boundary_ratio={args.teacher_boundary_ratio}",
    ]
    if args.vae_path:
        opts.append(f"model.config.tokenizer.vae_pth={args.vae_path}")
    if args.text_encoder_path:
        opts.append(f"model.config.text_encoder_path={args.text_encoder_path}")
    if args.neg_embed_path:
        opts.append(f"model.config.neg_embed_path={args.neg_embed_path}")
    if args.extra_overrides:
        opts.extend(args.extra_overrides)

    config = override(config, opts)
    model = instantiate(config.model)
    if not isinstance(model, T2VDistillModel_rCM):
        raise TypeError(f"Expected T2VDistillModel_rCM, got {type(model)}")
    model.eval()
    model.on_train_start()
    return model


def _student_init_consistency(model: T2VDistillModel_rCM) -> dict[str, Any]:
    expected = model._get_student_init_state_dict()
    actual = model.net.state_dict()

    num_compared_keys = 0
    num_skipped_keys = 0
    numel = 0
    max_abs = 0.0
    sum_abs = 0.0
    sum_diff_sq = 0.0
    sum_ref_sq = 0.0

    for key, expected_value in expected.items():
        actual_value = actual.get(key, None)
        if (
            isinstance(expected_value, torch.Tensor)
            and isinstance(actual_value, torch.Tensor)
            and expected_value.shape == actual_value.shape
            and expected_value.is_floating_point()
            and actual_value.is_floating_point()
        ):
            diff = (actual_value.float() - expected_value.float()).detach()
            if diff.numel() == 0:
                continue
            num_compared_keys += 1
            numel += diff.numel()
            max_abs = max(max_abs, float(diff.abs().max().item()))
            sum_abs += float(diff.abs().sum().item())
            sum_diff_sq += float(diff.pow(2).sum().item())
            sum_ref_sq += float(expected_value.float().pow(2).sum().item())
        else:
            num_skipped_keys += 1

    if numel == 0:
        raise RuntimeError("No floating-point parameters were compared for student init consistency.")

    mean_abs = sum_abs / numel
    rms_diff = (sum_diff_sq / numel) ** 0.5
    rms_ref = (sum_ref_sq / numel) ** 0.5
    rel_l2 = rms_diff / max(rms_ref, 1e-8)
    return {
        "num_compared_keys": num_compared_keys,
        "num_skipped_keys": num_skipped_keys,
        "num_compared_elements": numel,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_l2": rel_l2,
    }


def _student_fwitht_consistency(
    model: T2VDistillModel_rCM,
    *,
    dtype: torch.dtype,
    batch_size: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    time_min: float,
    time_max: float,
    dt_scale: float,
    eps: float,
) -> dict[str, Any]:
    in_dim = int(model.net.in_dim)
    text_len = int(model.net.text_len)
    text_dim = int(model.net.text_dim)

    xt = torch.randn(batch_size, in_dim, latent_t, latent_h, latent_w, device="cuda", dtype=dtype)
    dxt = torch.randn_like(xt)
    time = torch.rand(batch_size, 1, device="cuda", dtype=dtype) * (time_max - time_min) + time_min
    dtime = torch.randn_like(time) * dt_scale
    crossattn_emb = torch.randn(batch_size, text_len, text_dim, device="cuda", dtype=dtype)
    condition = TextCondition(crossattn_emb=crossattn_emb)

    with torch.no_grad():
        F_plain = model.denoise(xt, time, condition, net_type="student").F
    F_primal, F_tangent = model.student_F_withT(
        xt_B_C_T_H_W=(xt, dxt),
        time=(time, dtime),
        condition=condition,
    )
    primal_metrics = _metrics(F_plain, F_primal)

    with torch.no_grad():
        F_pos = model.denoise(xt + eps * dxt, time + eps * dtime, condition, net_type="student").F
        F_neg = model.denoise(xt - eps * dxt, time - eps * dtime, condition, net_type="student").F
    F_fd = (F_pos - F_neg) / (2.0 * eps)
    tangent_metrics = _metrics(F_fd, F_tangent)

    return {
        "primal_consistency": primal_metrics,
        "tangent_consistency": tangent_metrics,
    }


def _build_noise_labels(
    *,
    batch_size: int,
    threshold: float,
    t_scaling_factor: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    vals: list[float] = []
    if batch_size >= 1:
        vals.append(max(0.0, threshold - 5.0))
    if batch_size >= 2:
        vals.append(min(t_scaling_factor, threshold + 5.0))
    while len(vals) < batch_size:
        vals.append(float(torch.rand((), device="cpu").item() * t_scaling_factor))
    return torch.tensor(vals[:batch_size], device="cuda", dtype=dtype).view(batch_size, 1)


def _teacher_pipeline_consistency(
    model: T2VDistillModel_rCM,
    *,
    dtype: torch.dtype,
    batch_size: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    boundary_deltas: list[float],
) -> dict[str, Any]:
    if model.net_teacher_2 is None:
        raise RuntimeError("teacher_ckpt_2 is required for teacher pipeline preflight.")

    in_dim = int(model.net.in_dim)
    text_len = int(model.net.text_len)
    text_dim = int(model.net.text_dim)
    x = torch.randn(batch_size, in_dim, latent_t, latent_h, latent_w, device="cuda", dtype=dtype)
    crossattn_emb = torch.randn(batch_size, text_len, text_dim, device="cuda", dtype=dtype)
    condition = TextCondition(crossattn_emb=crossattn_emb)

    t_scaling = float(model.config.rectified_flow_t_scaling_factor)
    threshold = float(model.config.teacher_boundary_ratio * t_scaling)
    noise_labels = _build_noise_labels(
        batch_size=batch_size,
        threshold=threshold,
        t_scaling_factor=t_scaling,
        dtype=dtype,
    )
    rf_times = (noise_labels.double() / t_scaling).clamp(min=0.0, max=1.0)
    trig_time = rf_to_trig_time(rf_times).to(dtype=dtype)

    with torch.no_grad():
        F_dual = model.denoise(x, trig_time, condition, net_type="teacher").F

    time_B_1_T_1_1 = rearrange(trig_time, "b t -> b 1 t 1 1")
    c_skip, c_out, c_in, c_noise = model.scaling(time_B_1_T_1_1)
    x_input = (x * c_in).to(**model.tensor_kwargs)
    noise_input = c_noise.squeeze(dim=[1, 3, 4]).to(**model.tensor_kwargs)

    with torch.no_grad():
        out_1 = model._forward_net(model.net_teacher, x_input, noise_input, condition)
        out_2 = model._forward_net(model.net_teacher_2, x_input, noise_input, condition)
    mask_teacher_2 = model._teacher_2_mask_from_noise_labels(noise_input)
    expected_net = torch.where(mask_teacher_2.view(-1, 1, 1, 1, 1), out_2, out_1)
    x0_expected = c_skip * x + c_out * expected_net
    F_expected = (torch.cos(time_B_1_T_1_1) * x - x0_expected) / torch.sin(time_B_1_T_1_1)
    route_metrics = _metrics(F_expected, F_dual)

    jump_rows = []
    for delta in boundary_deltas:
        low_label = max(0.0, threshold - delta)
        high_label = min(t_scaling, threshold + delta)
        rf_low = torch.full((batch_size, 1), low_label / t_scaling, device="cuda", dtype=torch.float64)
        rf_high = torch.full((batch_size, 1), high_label / t_scaling, device="cuda", dtype=torch.float64)
        trig_low = rf_to_trig_time(rf_low).to(dtype=dtype)
        trig_high = rf_to_trig_time(rf_high).to(dtype=dtype)
        with torch.no_grad():
            F_low = model.denoise(x, trig_low, condition, net_type="teacher").F
            F_high = model.denoise(x, trig_high, condition, net_type="teacher").F
        jump_rows.append(
            {
                "delta_noise_label": float(delta),
                "low_noise_label": float(low_label),
                "high_noise_label": float(high_label),
                **_metrics(F_low, F_high),
            }
        )

    return {
        "threshold_noise_label": threshold,
        "teacher_2_hit_ratio": float(mask_teacher_2.float().mean().item()),
        "route_mismatch_metrics": route_metrics,
        "boundary_jump_metrics": jump_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training-state preflight for Wan2.2 A14B distillation prototype")
    parser.add_argument("--config", type=str, default="rcm/configs/registry_distill.py")
    parser.add_argument("--experiment", type=str, default="wan2pt2_a14b_res480p_t2v_scm")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--teacher_ckpt_2", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="")
    parser.add_argument("--text_encoder_path", type=str, default="")
    parser.add_argument("--neg_embed_path", type=str, default="")
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--teacher_init_strategy", type=str, default="average", choices=["teacher_1", "teacher_2", "average"])
    parser.add_argument("--teacher_init_low_noise_weight", type=float, default=0.5)
    parser.add_argument("--teacher_init_module_aware", action="store_true")
    parser.add_argument("--teacher_init_low_noise_weight_embed", type=float, default=0.5)
    parser.add_argument("--teacher_init_low_noise_weight_early", type=float, default=0.5)
    parser.add_argument("--teacher_init_low_noise_weight_late", type=float, default=0.5)
    parser.add_argument("--teacher_init_low_noise_weight_head", type=float, default=0.5)
    parser.add_argument("--teacher_boundary_ratio", type=float, default=0.875)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--latent_t", type=int, default=21)
    parser.add_argument("--latent_h", type=int, default=60)
    parser.add_argument("--latent_w", type=int, default=106)
    parser.add_argument("--time_min", type=float, default=0.2)
    parser.add_argument("--time_max", type=float, default=1.2)
    parser.add_argument("--dt_scale", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--boundary_deltas", type=str, default="1,5,10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--init_max_abs_threshold", type=float, default=1e-5)
    parser.add_argument("--init_rel_l2_threshold", type=float, default=1e-6)
    parser.add_argument("--primal_max_abs_threshold", type=float, default=1e-4)
    parser.add_argument("--primal_rel_l2_threshold", type=float, default=1e-4)
    parser.add_argument("--tangent_max_abs_threshold", type=float, default=1e-3)
    parser.add_argument("--tangent_rel_l2_threshold", type=float, default=5e-3)
    parser.add_argument("--route_mismatch_max_abs_threshold", type=float, default=1e-6)
    parser.add_argument("--route_mismatch_rel_l2_threshold", type=float, default=1e-6)
    parser.add_argument("--boundary_jump_rel_l2_warn_threshold", type=float, default=2.0)
    parser.add_argument("--extra_overrides", type=str, nargs="*", default=[])
    parser.add_argument("--save_json", type=str, default="")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training-state preflight.")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {args.batch_size}")
    if not (0.0 <= args.teacher_init_low_noise_weight <= 1.0):
        raise ValueError("--teacher_init_low_noise_weight must be in [0, 1].")
    for name in (
        "teacher_init_low_noise_weight_embed",
        "teacher_init_low_noise_weight_early",
        "teacher_init_low_noise_weight_late",
        "teacher_init_low_noise_weight_head",
    ):
        value = float(getattr(args, name))
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"--{name} must be in [0, 1], got {value}")

    dtype = _resolve_dtype(args.precision)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = _load_model_from_config(args)

    init_metrics = _student_init_consistency(model)
    fwitht_metrics = _student_fwitht_consistency(
        model,
        dtype=dtype,
        batch_size=args.batch_size,
        latent_t=args.latent_t,
        latent_h=args.latent_h,
        latent_w=args.latent_w,
        time_min=args.time_min,
        time_max=args.time_max,
        dt_scale=args.dt_scale,
        eps=args.eps,
    )
    pipeline_metrics = _teacher_pipeline_consistency(
        model,
        dtype=dtype,
        batch_size=args.batch_size,
        latent_t=args.latent_t,
        latent_h=args.latent_h,
        latent_w=args.latent_w,
        boundary_deltas=_parse_csv_floats(args.boundary_deltas),
    )

    report = {
        "config": {
            "config": args.config,
            "experiment": args.experiment,
            "precision": args.precision,
            "teacher_init_strategy": args.teacher_init_strategy,
            "teacher_init_low_noise_weight": args.teacher_init_low_noise_weight,
            "teacher_init_module_aware": bool(args.teacher_init_module_aware),
            "teacher_init_low_noise_weight_embed": args.teacher_init_low_noise_weight_embed,
            "teacher_init_low_noise_weight_early": args.teacher_init_low_noise_weight_early,
            "teacher_init_low_noise_weight_late": args.teacher_init_low_noise_weight_late,
            "teacher_init_low_noise_weight_head": args.teacher_init_low_noise_weight_head,
            "teacher_boundary_ratio": args.teacher_boundary_ratio,
            "batch_size": args.batch_size,
            "latent_shape": [args.latent_t, args.latent_h, args.latent_w],
            "time_range": [args.time_min, args.time_max],
            "dt_scale": args.dt_scale,
            "eps": args.eps,
            "boundary_deltas": args.boundary_deltas,
            "seed": args.seed,
        },
        "student_init_consistency": init_metrics,
        "student_fwitht_consistency": fwitht_metrics,
        "teacher_pipeline_consistency": pipeline_metrics,
    }

    print(
        "[init] "
        f"max_abs={init_metrics['max_abs']:.6e}, "
        f"mean_abs={init_metrics['mean_abs']:.6e}, "
        f"rel_l2={init_metrics['rel_l2']:.6e}"
    )
    print(
        "[student_fwitht.primal] "
        f"max_abs={fwitht_metrics['primal_consistency']['max_abs']:.6e}, "
        f"rel_l2={fwitht_metrics['primal_consistency']['rel_l2']:.6e}"
    )
    print(
        "[student_fwitht.tangent] "
        f"max_abs={fwitht_metrics['tangent_consistency']['max_abs']:.6e}, "
        f"rel_l2={fwitht_metrics['tangent_consistency']['rel_l2']:.6e}"
    )
    print(
        "[teacher.route] "
        f"teacher_2_hit_ratio={pipeline_metrics['teacher_2_hit_ratio']:.3f}, "
        f"mismatch_max_abs={pipeline_metrics['route_mismatch_metrics']['max_abs']:.6e}, "
        f"mismatch_rel_l2={pipeline_metrics['route_mismatch_metrics']['rel_l2']:.6e}"
    )
    for row in pipeline_metrics["boundary_jump_metrics"]:
        print(
            "[teacher.boundary_jump] "
            f"delta={row['delta_noise_label']:.3f}, "
            f"mean_abs={row['mean_abs']:.6e}, "
            f"rel_l2={row['rel_l2']:.6e}"
        )

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[save] wrote training-state preflight report to {output_path}")

    if args.strict:
        if init_metrics["max_abs"] > args.init_max_abs_threshold or init_metrics["rel_l2"] > args.init_rel_l2_threshold:
            raise RuntimeError(
                "Student init consistency check failed: "
                f"max_abs={init_metrics['max_abs']:.6e} (thr={args.init_max_abs_threshold:.6e}), "
                f"rel_l2={init_metrics['rel_l2']:.6e} (thr={args.init_rel_l2_threshold:.6e})"
            )

        primal = fwitht_metrics["primal_consistency"]
        if primal["max_abs"] > args.primal_max_abs_threshold or primal["rel_l2"] > args.primal_rel_l2_threshold:
            raise RuntimeError(
                "student_F_withT primal consistency failed: "
                f"max_abs={primal['max_abs']:.6e} (thr={args.primal_max_abs_threshold:.6e}), "
                f"rel_l2={primal['rel_l2']:.6e} (thr={args.primal_rel_l2_threshold:.6e})"
            )

        tangent = fwitht_metrics["tangent_consistency"]
        if tangent["max_abs"] > args.tangent_max_abs_threshold or tangent["rel_l2"] > args.tangent_rel_l2_threshold:
            raise RuntimeError(
                "student_F_withT tangent consistency failed: "
                f"max_abs={tangent['max_abs']:.6e} (thr={args.tangent_max_abs_threshold:.6e}), "
                f"rel_l2={tangent['rel_l2']:.6e} (thr={args.tangent_rel_l2_threshold:.6e})"
            )

        route = pipeline_metrics["route_mismatch_metrics"]
        if route["max_abs"] > args.route_mismatch_max_abs_threshold or route["rel_l2"] > args.route_mismatch_rel_l2_threshold:
            raise RuntimeError(
                "Teacher pipeline route mismatch failed: "
                f"max_abs={route['max_abs']:.6e} (thr={args.route_mismatch_max_abs_threshold:.6e}), "
                f"rel_l2={route['rel_l2']:.6e} (thr={args.route_mismatch_rel_l2_threshold:.6e})"
            )

        max_boundary_rel_l2 = max(row["rel_l2"] for row in pipeline_metrics["boundary_jump_metrics"])
        if max_boundary_rel_l2 > args.boundary_jump_rel_l2_warn_threshold:
            raise RuntimeError(
                "Teacher boundary-jump check failed: "
                f"max_rel_l2={max_boundary_rel_l2:.6e} (thr={args.boundary_jump_rel_l2_warn_threshold:.6e})"
            )

    print("[done] training-state preflight completed")


if __name__ == "__main__":
    main()
