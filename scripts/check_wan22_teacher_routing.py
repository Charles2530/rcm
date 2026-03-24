#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect dual-teacher boundary routing behavior for Wan2.2 prototype training.

This mirrors the current routing rule used in training:
    teacher_2 if noise_label <= teacher_boundary_ratio * rectified_flow_t_scaling_factor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _parse_csv_floats(csv: str) -> list[float]:
    vals = [s.strip() for s in csv.split(",") if s.strip()]
    return [float(v) for v in vals]


def _route_mask(noise_labels: torch.Tensor, boundary_ratio: float, t_scaling_factor: float) -> torch.Tensor:
    threshold = boundary_ratio * t_scaling_factor
    return noise_labels <= threshold


def _analyze_single_ratio(
    *,
    ratio: float,
    t_scaling_factor: float,
    noise_labels: torch.Tensor,
    rf_times: torch.Tensor,
    neighborhood_delta: float,
    random_noise_labels: torch.Tensor,
) -> dict[str, Any]:
    threshold = ratio * t_scaling_factor

    mask_noise = _route_mask(noise_labels=noise_labels, boundary_ratio=ratio, t_scaling_factor=t_scaling_factor)
    mask_rf = _route_mask(noise_labels=rf_times * t_scaling_factor, boundary_ratio=ratio, t_scaling_factor=t_scaling_factor)
    mask_random = _route_mask(noise_labels=random_noise_labels, boundary_ratio=ratio, t_scaling_factor=t_scaling_factor)

    around = torch.tensor(
        [
            threshold - neighborhood_delta,
            threshold - neighborhood_delta / 2.0,
            threshold,
            threshold + neighborhood_delta / 2.0,
            threshold + neighborhood_delta,
        ],
        dtype=torch.float64,
    )
    mask_around = _route_mask(noise_labels=around, boundary_ratio=ratio, t_scaling_factor=t_scaling_factor)

    return {
        "boundary_ratio": ratio,
        "threshold_noise_label": threshold,
        "noise_label_cases": [
            {"noise_label": float(v), "route_to_teacher_2": bool(m)} for v, m in zip(noise_labels.tolist(), mask_noise.tolist())
        ],
        "rf_time_cases": [{"rf_time": float(v), "route_to_teacher_2": bool(m)} for v, m in zip(rf_times.tolist(), mask_rf.tolist())],
        "teacher2_hit_ratio_noise_labels": float(mask_noise.float().mean().item()) if mask_noise.numel() > 0 else 0.0,
        "teacher2_hit_ratio_rf_times": float(mask_rf.float().mean().item()) if mask_rf.numel() > 0 else 0.0,
        "teacher2_hit_ratio_uniform_random": float(mask_random.float().mean().item()) if mask_random.numel() > 0 else 0.0,
        "boundary_neighborhood": [
            {"noise_label": float(v), "route_to_teacher_2": bool(m)} for v, m in zip(around.tolist(), mask_around.tolist())
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check boundary-based dual-teacher routing statistics")
    parser.add_argument("--boundary_ratios", type=str, default="0.75,0.8,0.875,0.9")
    parser.add_argument("--rectified_flow_t_scaling_factor", type=float, default=1000.0)
    parser.add_argument("--noise_labels", type=str, default="0,50,250,500,750,875,900,999,1000")
    parser.add_argument("--rf_times", type=str, default="0.0,0.05,0.25,0.5,0.75,0.875,0.9,0.999,1.0")
    parser.add_argument("--num_random_noise_labels", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighborhood_delta", type=float, default=5.0)
    parser.add_argument("--save_json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = _parse_csv_floats(args.boundary_ratios)
    noise_labels = torch.tensor(_parse_csv_floats(args.noise_labels), dtype=torch.float64)
    rf_times = torch.tensor(_parse_csv_floats(args.rf_times), dtype=torch.float64)
    if torch.any(rf_times < 0.0) or torch.any(rf_times > 1.0):
        raise ValueError("rf_times must be within [0, 1].")

    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed)
    random_noise_labels = torch.rand(args.num_random_noise_labels, generator=g, dtype=torch.float64) * args.rectified_flow_t_scaling_factor

    results = []
    print("[routing-summary] boundary-based teacher_2 hit ratios")
    for ratio in ratios:
        info = _analyze_single_ratio(
            ratio=ratio,
            t_scaling_factor=args.rectified_flow_t_scaling_factor,
            noise_labels=noise_labels,
            rf_times=rf_times,
            neighborhood_delta=args.neighborhood_delta,
            random_noise_labels=random_noise_labels,
        )
        results.append(info)
        print(
            f"  - ratio={ratio:.6f}, threshold={info['threshold_noise_label']:.3f}, "
            f"hit(noise_labels)={info['teacher2_hit_ratio_noise_labels']:.3f}, "
            f"hit(rf_times)={info['teacher2_hit_ratio_rf_times']:.3f}, "
            f"hit(uniform_random)={info['teacher2_hit_ratio_uniform_random']:.3f}"
        )

    payload = {
        "config": {
            "boundary_ratios": ratios,
            "rectified_flow_t_scaling_factor": args.rectified_flow_t_scaling_factor,
            "noise_labels": noise_labels.tolist(),
            "rf_times": rf_times.tolist(),
            "num_random_noise_labels": args.num_random_noise_labels,
            "seed": args.seed,
            "neighborhood_delta": args.neighborhood_delta,
        },
        "results": results,
    }

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[save] wrote routing report to {output_path}")

    print("[done] routing analysis completed")


if __name__ == "__main__":
    main()
