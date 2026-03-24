#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression parity check for Wan2.2 T2V A14B teacher conversion.

Compares Diffusers experts (`transformer`, `transformer_2`) against converted
rCM checkpoints across multiple seeds / timesteps / latent shapes / dtypes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

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


def _parse_csv_ints(csv: str) -> list[int]:
    vals = [s.strip() for s in csv.split(",") if s.strip()]
    return [int(v) for v in vals]


def _parse_csv_strs(csv: str) -> list[str]:
    return [s.strip() for s in csv.split(",") if s.strip()]


def _parse_shape_tokens(csv: str) -> list[tuple[int, int, int]]:
    out = []
    for token in _parse_csv_strs(csv):
        parts = token.lower().split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid latent shape token '{token}', expected 'T×H×W', e.g. 21x60x106")
        out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return out


def _build_case_iter(
    seeds: Iterable[int], timesteps: Iterable[int], latent_shapes: Iterable[tuple[int, int, int]]
) -> Iterable[tuple[int, int, tuple[int, int, int]]]:
    for seed in seeds:
        for timestep in timesteps:
            for shape in latent_shapes:
                yield seed, timestep, shape


def _case_metrics(out_hf: torch.Tensor, out_rcm: torch.Tensor) -> dict[str, float]:
    diff = out_hf.float() - out_rcm.float()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel_l2 = (diff.pow(2).mean().sqrt() / out_hf.float().pow(2).mean().sqrt().clamp(min=1e-8)).item()
    return {"max_abs": max_abs, "mean_abs": mean_abs, "rel_l2": rel_l2}


@torch.no_grad()
def _run_expert_parity(
    *,
    model_root: Path,
    converted_ckpt: Path,
    subdir: str,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    prefix: str,
    seeds: list[int],
    timesteps: list[int],
    latent_shapes: list[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    transformer_dir = model_root / subdir
    cfg = _load_json(transformer_dir / "config.json")
    net_args = _parse_wan22_t2v_config(cfg)

    WanTransformer3DModel = _import_wan_transformer_cls()
    hf_model = WanTransformer3DModel.from_pretrained(str(transformer_dir), torch_dtype=dtype, local_files_only=True).to(
        device=device, dtype=dtype
    )
    hf_model.eval()

    rcm_model = WanModel(**net_args).to(device=device, dtype=dtype)
    rcm_model.eval()

    converted_state = _strip_prefix(load_state_dict(str(converted_ckpt)), prefix=prefix)
    incompatible = rcm_model.load_state_dict(converted_state, strict=False)
    print(f"[{subdir}/{dtype}] missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}")

    rows: list[dict[str, Any]] = []
    for seed, timestep, (latent_t, latent_h, latent_w) in _build_case_iter(seeds=seeds, timesteps=timesteps, latent_shapes=latent_shapes):
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        x = torch.randn(batch_size, net_args["in_dim"], latent_t, latent_h, latent_w, device=device, dtype=dtype, generator=g)
        t = torch.full((batch_size,), float(timestep), device=device, dtype=dtype)
        crossattn_emb = torch.randn(batch_size, net_args["text_len"], net_args["text_dim"], device=device, dtype=dtype, generator=g)

        out_hf = _forward_diffusers(hf_model, x, t, crossattn_emb)
        out_rcm = rcm_model(
            x_B_C_T_H_W=x,
            timesteps_B_T=t.view(batch_size, 1),
            crossattn_emb=crossattn_emb,
        )

        if out_hf.shape != out_rcm.shape:
            raise RuntimeError(
                f"[{subdir}/{dtype}] shape mismatch at seed={seed}, t={timestep}, shape={latent_t}x{latent_h}x{latent_w}: "
                f"diffusers={tuple(out_hf.shape)}, rcm={tuple(out_rcm.shape)}"
            )

        metrics = _case_metrics(out_hf=out_hf, out_rcm=out_rcm)
        row = {
            "expert": subdir,
            "dtype": str(dtype).replace("torch.", ""),
            "seed": seed,
            "timestep": timestep,
            "latent_shape": [latent_t, latent_h, latent_w],
            **metrics,
        }
        rows.append(row)

    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['expert']}/{row['dtype']}"
        grouped.setdefault(key, []).append(row)

    summary: dict[str, dict[str, float]] = {}
    for key, group_rows in grouped.items():
        max_abs_vals = [float(r["max_abs"]) for r in group_rows]
        mean_abs_vals = [float(r["mean_abs"]) for r in group_rows]
        rel_l2_vals = [float(r["rel_l2"]) for r in group_rows]
        summary[key] = {
            "num_cases": len(group_rows),
            "max_abs_max": max(max_abs_vals),
            "mean_abs_mean": float(sum(mean_abs_vals) / len(mean_abs_vals)),
            "rel_l2_mean": float(sum(rel_l2_vals) / len(rel_l2_vals)),
            "rel_l2_max": max(rel_l2_vals),
        }
    return summary


def _strict_failures(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        reasons = []
        if row["max_abs"] > args.max_abs_threshold:
            reasons.append(f"max_abs>{args.max_abs_threshold}")
        if row["mean_abs"] > args.mean_abs_threshold:
            reasons.append(f"mean_abs>{args.mean_abs_threshold}")
        if row["rel_l2"] > args.rel_l2_threshold:
            reasons.append(f"rel_l2>{args.rel_l2_threshold}")
        if reasons:
            failures.append({**row, "reasons": reasons})
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression parity check for Wan2.2 teacher conversion")
    parser.add_argument("--model_root", type=str, default="./model/Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument("--converted_root", type=str, default="./model/Wan2.2-T2V-A14B-Diffusers-rcm")
    parser.add_argument("--ckpt_transformer", type=str, default="Wan2.2-T2V-A14B-transformer-rcm.pth")
    parser.add_argument("--ckpt_transformer_2", type=str, default="Wan2.2-T2V-A14B-transformer_2-rcm.pth")
    parser.add_argument("--skip_transformer_2", action="store_true")
    parser.add_argument("--prefix", type=str, default="net.")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dtypes", type=str, default="float32,bfloat16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--timesteps", type=str, default="0,50,250,500,750,999")
    parser.add_argument("--latent_shapes", type=str, default="21x60x106,21x48x84")
    parser.add_argument("--strict", action="store_true", help="Fail when any case exceeds thresholds.")
    parser.add_argument("--max_abs_threshold", type=float, default=1e-3)
    parser.add_argument("--mean_abs_threshold", type=float, default=1e-4)
    parser.add_argument("--rel_l2_threshold", type=float, default=1e-3)
    parser.add_argument("--save_json", type=str, default="", help="Optional path to save full regression results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_root = Path(args.model_root)
    converted_root = Path(args.converted_root)
    dtype_names = _parse_csv_strs(args.dtypes)
    seeds = _parse_csv_ints(args.seeds)
    timesteps = _parse_csv_ints(args.timesteps)
    latent_shapes = _parse_shape_tokens(args.latent_shapes)

    experts: list[tuple[str, Path]] = [("transformer", converted_root / args.ckpt_transformer)]
    if not args.skip_transformer_2:
        experts.append(("transformer_2", converted_root / args.ckpt_transformer_2))

    rows: list[dict[str, Any]] = []
    skipped_dtypes: list[str] = []
    for dtype_name in dtype_names:
        dtype = _resolve_dtype(dtype_name)
        if args.device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
            skipped_dtypes.append(dtype_name)
            print(f"[skip] dtype={dtype_name} on CPU is skipped.")
            continue

        for subdir, ckpt_path in experts:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Converted checkpoint does not exist: {ckpt_path}")
            rows.extend(
                _run_expert_parity(
                    model_root=model_root,
                    converted_ckpt=ckpt_path,
                    subdir=subdir,
                    device=args.device,
                    dtype=dtype,
                    batch_size=args.batch_size,
                    prefix=args.prefix,
                    seeds=seeds,
                    timesteps=timesteps,
                    latent_shapes=latent_shapes,
                )
            )

    summary = _summarize(rows)
    failures = _strict_failures(rows, args) if args.strict else []

    print("[summary] parity regression")
    for key, stats in summary.items():
        print(
            f"  - {key}: cases={int(stats['num_cases'])}, "
            f"max_abs_max={stats['max_abs_max']:.6e}, "
            f"mean_abs_mean={stats['mean_abs_mean']:.6e}, "
            f"rel_l2_mean={stats['rel_l2_mean']:.6e}, "
            f"rel_l2_max={stats['rel_l2_max']:.6e}"
        )
    if skipped_dtypes:
        print(f"[summary] skipped dtypes on {args.device}: {', '.join(skipped_dtypes)}")

    if args.save_json:
        output = {
            "args": vars(args),
            "summary": summary,
            "num_results": len(rows),
            "results": rows,
            "num_failures": len(failures),
            "failures": failures,
        }
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"[save] wrote regression report to {output_path}")

    if args.strict and failures:
        first = failures[0]
        raise RuntimeError(
            f"Parity strict check failed with {len(failures)} failing cases. "
            f"First failure: expert={first['expert']} dtype={first['dtype']} seed={first['seed']} "
            f"t={first['timestep']} shape={first['latent_shape']} reasons={first['reasons']}"
        )

    print("[done] parity regression completed")


if __name__ == "__main__":
    main()
