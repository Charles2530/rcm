#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Wan2.2 T2V A14B Diffusers weights to RCM-compatible checkpoints.

This script converts Diffusers `WanTransformer3DModel` checkpoints from:
- `<model_root>/transformer`
- `<model_root>/transformer_2` (optional low-noise expert)

to standalone `.pth` files with `net.*`-prefixed keys that can be loaded by
`rcm/models/t2v_model_distill_rcm.py`.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Mapping, Optional, Tuple

import torch

from rcm.networks.wan2pt2 import WanModel
from rcm.utils.model_utils import load_state_dict_from_folder


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_wan22_t2v_config(transformer_config: Mapping[str, object]) -> dict:
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


def _map_attention_key(block_id: str, suffix: str, attn_name: str) -> Optional[str]:
    attn_map = {
        "to_q.weight": "q.weight",
        "to_q.bias": "q.bias",
        "to_k.weight": "k.weight",
        "to_k.bias": "k.bias",
        "to_v.weight": "v.weight",
        "to_v.bias": "v.bias",
        "to_out.0.weight": "o.weight",
        "to_out.0.bias": "o.bias",
        "norm_q.weight": "norm_q.weight",
        "norm_k.weight": "norm_k.weight",
    }
    if suffix not in attn_map:
        return None
    return f"blocks.{block_id}.{attn_name}.{attn_map[suffix]}"


def _map_hf_key_to_rcm(hf_key: str) -> Optional[str]:
    if hf_key.startswith("blocks."):
        # Example: blocks.0.attn1.to_q.weight
        parts = hf_key.split(".")
        if len(parts) < 4:
            return None
        block_id = parts[1]
        suffix = ".".join(parts[2:])

        if suffix.startswith("attn1."):
            mapped = _map_attention_key(block_id, suffix[len("attn1.") :], "self_attn")
            if mapped is not None:
                return mapped
        if suffix.startswith("attn2."):
            mapped = _map_attention_key(block_id, suffix[len("attn2.") :], "cross_attn")
            if mapped is not None:
                return mapped

        if suffix.startswith("ffn.net.0.proj."):
            return f"blocks.{block_id}.ffn.0.{suffix[len('ffn.net.0.proj.') :]}"
        if suffix.startswith("ffn.net.2."):
            return f"blocks.{block_id}.ffn.2.{suffix[len('ffn.net.2.') :]}"
        if suffix.startswith("norm2."):
            return f"blocks.{block_id}.norm3.{suffix[len('norm2.') :]}"
        if suffix == "scale_shift_table":
            return f"blocks.{block_id}.modulation"

        return None

    top_level = {
        "patch_embedding.weight": "patch_embedding.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
        "condition_embedder.time_proj.weight": "time_projection.1.weight",
        "condition_embedder.time_proj.bias": "time_projection.1.bias",
        "proj_out.weight": "head.head.weight",
        "proj_out.bias": "head.head.bias",
        "scale_shift_table": "head.modulation",
    }
    return top_level.get(hf_key)


def _convert_transformer_state_dict(hf_state_dict: Mapping[str, torch.Tensor]) -> Tuple[collections.OrderedDict, list[str]]:
    converted = collections.OrderedDict()
    unmapped = []

    for key, value in hf_state_dict.items():
        mapped_key = _map_hf_key_to_rcm(key)
        if mapped_key is None:
            unmapped.append(key)
            continue
        if mapped_key.endswith(".modulation") and isinstance(value, torch.Tensor) and value.ndim == 2:
            # Diffusers stores modulation as [N, D], rCM expects [1, N, D].
            value = value.unsqueeze(0)
        converted[mapped_key] = value

    return converted, sorted(unmapped)


def _validate_converted_keys(
    converted: Mapping[str, torch.Tensor],
    net_args: Mapping[str, object],
    strict: bool,
) -> None:
    target_state = WanModel(**net_args).state_dict()

    converted_keys = set(converted.keys())
    target_keys = set(target_state.keys())

    missing = sorted(target_keys - converted_keys)
    unexpected = sorted(converted_keys - target_keys)

    shape_mismatch = []
    for k in sorted(converted_keys & target_keys):
        if tuple(converted[k].shape) != tuple(target_state[k].shape):
            shape_mismatch.append((k, tuple(converted[k].shape), tuple(target_state[k].shape)))

    print(f"[validate] converted={len(converted_keys)} target={len(target_keys)}")
    print(f"[validate] missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape_mismatch)}")

    if missing[:10]:
        print("[validate] first missing keys:")
        for k in missing[:10]:
            print("  -", k)
    if unexpected[:10]:
        print("[validate] first unexpected keys:")
        for k in unexpected[:10]:
            print("  -", k)
    if shape_mismatch[:10]:
        print("[validate] first shape mismatches:")
        for k, src_shape, dst_shape in shape_mismatch[:10]:
            print(f"  - {k}: src={src_shape} dst={dst_shape}")

    if strict and (missing or unexpected or shape_mismatch):
        raise RuntimeError("Validation failed in strict mode.")


def _save_prefixed_checkpoint(state_dict: Mapping[str, torch.Tensor], output_path: Path, prefix: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prefixed = collections.OrderedDict((f"{prefix}{k}", v.cpu()) for k, v in state_dict.items())
    torch.save(prefixed, output_path)
    print(f"[save] {output_path}")


def _convert_one_expert(
    model_root: Path,
    subdir: str,
    output_path: Path,
    prefix: str,
    strict: bool,
) -> dict:
    transformer_dir = model_root / subdir
    transformer_cfg_path = transformer_dir / "config.json"

    if not transformer_dir.is_dir():
        raise FileNotFoundError(f"Cannot find transformer dir: {transformer_dir}")
    if not transformer_cfg_path.exists():
        raise FileNotFoundError(f"Cannot find config.json: {transformer_cfg_path}")

    transformer_cfg = _load_json(transformer_cfg_path)
    net_args = _parse_wan22_t2v_config(transformer_cfg)

    print(f"[load] reading weights from {transformer_dir}")
    hf_state = load_state_dict_from_folder(str(transformer_dir))
    converted, unmapped = _convert_transformer_state_dict(hf_state)
    print(f"[map] mapped={len(converted)} unmapped={len(unmapped)}")
    if unmapped[:20]:
        print("[map] first unmapped keys:")
        for k in unmapped[:20]:
            print("  -", k)

    _validate_converted_keys(converted, net_args, strict=strict)
    _save_prefixed_checkpoint(converted, output_path=output_path, prefix=prefix)

    return {
        "subdir": subdir,
        "output": str(output_path),
        "mapped": len(converted),
        "unmapped": len(unmapped),
        "config": net_args,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Wan2.2 T2V A14B Diffusers checkpoints to RCM format")
    parser.add_argument(
        "--model_root",
        type=str,
        default="./model/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Path to local diffusers model root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model/Wan-AI/Wan2.2-T2V-A14B-Diffusers-rcm",
        help="Output directory for converted checkpoints.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="net.",
        help="Prefix added to every converted key (default: net.).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if key/shape validation is not exact.",
    )
    parser.add_argument(
        "--skip_transformer_2",
        action="store_true",
        help="Skip conversion for transformer_2 (low-noise expert).",
    )
    parser.add_argument(
        "--output_name_transformer",
        type=str,
        default="Wan2.2-T2V-A14B-transformer-rcm.pth",
        help="Filename for converted transformer expert.",
    )
    parser.add_argument(
        "--output_name_transformer_2",
        type=str,
        default="Wan2.2-T2V-A14B-transformer_2-rcm.pth",
        help="Filename for converted transformer_2 expert.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = Path(args.model_root)
    output_dir = Path(args.output_dir)

    if not model_root.is_dir():
        raise FileNotFoundError(f"model_root does not exist: {model_root}")

    reports = []

    reports.append(
        _convert_one_expert(
            model_root=model_root,
            subdir="transformer",
            output_path=output_dir / args.output_name_transformer,
            prefix=args.prefix,
            strict=args.strict,
        )
    )

    if not args.skip_transformer_2:
        reports.append(
            _convert_one_expert(
                model_root=model_root,
                subdir="transformer_2",
                output_path=output_dir / args.output_name_transformer_2,
                prefix=args.prefix,
                strict=args.strict,
            )
        )

    report_path = output_dir / "conversion_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print(f"[done] report written to {report_path}")


if __name__ == "__main__":
    main()
