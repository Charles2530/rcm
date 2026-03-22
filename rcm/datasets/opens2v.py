# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import glob
import json
import os
import random
from collections.abc import Iterator, Mapping
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from imaginaire.utils import log
from rcm.datasets.utils import VIDEO_RES_SIZE_INFO

try:
    from torchvision.io import read_video
except Exception:
    read_video = None

try:
    import imageio.v3 as iio
except Exception:
    iio = None


def dict_collation_fn(samples):
    if not samples:
        return {}

    keys = samples[0].keys()
    batched_dict = {key: [] for key in keys}

    for sample in samples:
        for key in keys:
            batched_dict[key].append(sample[key])

    for key in keys:
        if isinstance(batched_dict[key][0], torch.Tensor):
            batched_dict[key] = torch.stack(batched_dict[key])

    return batched_dict


def _parse_target_hw(target_resolution: str, target_aspect_ratio: str) -> tuple[int, int]:
    if target_resolution in VIDEO_RES_SIZE_INFO and target_aspect_ratio in VIDEO_RES_SIZE_INFO[target_resolution]:
        w, h = VIDEO_RES_SIZE_INFO[target_resolution][target_aspect_ratio]
        return h, w

    # fallback: explicit "WxH"
    if "x" in target_resolution.lower():
        w_str, h_str = target_resolution.lower().split("x")
        return int(h_str), int(w_str)

    raise ValueError(
        f"Unsupported target resolution/aspect ratio: {target_resolution} {target_aspect_ratio}. "
        f"Supported presets: {list(VIDEO_RES_SIZE_INFO.keys())}"
    )


def _iter_json_entries(json_path: str) -> Iterator[tuple[str, Any]]:
    # For large OpenS2V json files, use streaming parser if available.
    try:
        import ijson  # type: ignore

        with open(json_path, "rb") as f:
            for key, value in ijson.kvitems(f, ""):
                yield str(key), value
        return
    except Exception:
        pass

    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            yield str(key), value
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield str(idx), value
    else:
        raise ValueError(f"Unsupported JSON root type in {json_path}: {type(obj)}")


class OpenS2VIterableDataset(IterableDataset):
    def __init__(
        self,
        index_json_pattern: str,
        videos_root: str,
        target_resolution: str = "480p",
        target_aspect_ratio: str = "16:9",
        max_frames: int = 81,
        decoder: str = "torchvision",
        use_face_cut: bool = True,
        use_crop: bool = True,
        max_entries_per_json: int = -1,
        skip_broken_samples: bool = True,
        shuffle_json_files: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.index_json_pattern = index_json_pattern
        self.videos_root = videos_root
        self.target_h, self.target_w = _parse_target_hw(target_resolution, target_aspect_ratio)
        self.max_frames = max_frames
        self.decoder = decoder
        self.use_face_cut = use_face_cut
        self.use_crop = use_crop
        self.max_entries_per_json = max_entries_per_json
        self.skip_broken_samples = skip_broken_samples
        self.shuffle_json_files = shuffle_json_files
        self.seed = seed

    def _decode_video_tchw_uint8(self, video_path: str) -> torch.Tensor:
        last_err = None

        if self.decoder in {"torchvision", "auto"} and read_video is not None:
            try:
                video_tchw, _, _ = read_video(video_path, pts_unit="sec", output_format="TCHW")
                if isinstance(video_tchw, torch.Tensor) and video_tchw.numel() > 0:
                    return video_tchw.to(torch.uint8)
            except Exception as e:
                last_err = e

        if self.decoder in {"imageio", "auto", "torchvision"} and iio is not None:
            try:
                frames = [torch.from_numpy(frame) for frame in iio.imiter(video_path)]
                if frames:
                    video = torch.stack(frames, dim=0)  # [T, H, W, C]
                    return video.permute(0, 3, 1, 2).contiguous().to(torch.uint8)
            except Exception as e:
                last_err = e

        raise RuntimeError(f"Failed to decode video: {video_path}. Last error: {last_err}")

    def _extract_metadata(self, entry: Any) -> Mapping[str, Any]:
        if isinstance(entry, Mapping) and isinstance(entry.get("metadata"), Mapping):
            return entry["metadata"]
        if isinstance(entry, Mapping):
            return entry
        raise ValueError(f"Unexpected OpenS2V entry type: {type(entry)}")

    def _extract_caption(self, meta: Mapping[str, Any], entry: Mapping[str, Any]) -> str:
        for key in ["cap", "caption", "text", "prompt"]:
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value and isinstance(value[0], str) and value[0].strip():
                return value[0].strip()

        for key in ["cap", "caption", "text", "prompt"]:
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value and isinstance(value[0], str) and value[0].strip():
                return value[0].strip()

        raise KeyError("No caption field found in OpenS2V entry.")

    def _extract_video_relpath(self, meta: Mapping[str, Any], entry: Mapping[str, Any]) -> str:
        for key in ["path", "video_path", "video", "video_relpath"]:
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for key in ["path", "video_path", "video", "video_relpath"]:
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise KeyError("No video path field found in OpenS2V entry.")

    def _resolve_video_path(self, relpath: str) -> str:
        relpath = relpath.lstrip("/")
        path = os.path.join(self.videos_root, relpath)
        if os.path.isfile(path):
            return path

        # Common extraction layout fallback: Videos/total_partX/.../part_xxxx/...
        matches = glob.glob(os.path.join(self.videos_root, "**", relpath), recursive=True)
        if matches:
            return matches[0]

        raise FileNotFoundError(f"Cannot resolve video path: relpath={relpath}, videos_root={self.videos_root}")

    def _apply_face_cut_and_crop(self, video_tchw: torch.Tensor, meta: Mapping[str, Any]) -> torch.Tensor:
        out = video_tchw

        if self.use_face_cut:
            face_cut = meta.get("face_cut") or meta.get("cut")
            if isinstance(face_cut, (list, tuple)) and len(face_cut) >= 2:
                s = max(0, int(face_cut[0]))
                e = max(s + 1, int(face_cut[1]))
                s = min(s, out.shape[0] - 1)
                e = min(e, out.shape[0])
                out = out[s:e]

        if self.use_crop:
            crop = meta.get("crop")
            if isinstance(crop, (list, tuple)) and len(crop) >= 4:
                s_x, e_x, s_y, e_y = [int(v) for v in crop[:4]]
                h, w = out.shape[2], out.shape[3]
                s_x = max(0, min(s_x, w - 1))
                e_x = max(s_x + 1, min(e_x, w))
                s_y = max(0, min(s_y, h - 1))
                e_y = max(s_y + 1, min(e_y, h))
                out = out[:, :, s_y:e_y, s_x:e_x]

        return out

    def _resize_video_tchw_uint8(self, video_tchw: torch.Tensor) -> torch.Tensor:
        if video_tchw.shape[2] == self.target_h and video_tchw.shape[3] == self.target_w:
            return video_tchw

        resized = F.interpolate(
            video_tchw.float(),
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return resized.round().clamp_(0, 255).to(torch.uint8)

    def _uniform_trim_frames(self, video_tchw: torch.Tensor) -> torch.Tensor:
        if self.max_frames <= 0 or video_tchw.shape[0] <= self.max_frames:
            return video_tchw
        idx = torch.linspace(0, video_tchw.shape[0] - 1, self.max_frames, dtype=torch.float32)
        idx = idx.round().long().clamp_(0, video_tchw.shape[0] - 1)
        return video_tchw.index_select(0, idx)

    def _build_sample(self, sample_id: str, entry: Any) -> dict[str, Any]:
        if not isinstance(entry, Mapping):
            raise ValueError(f"Sample {sample_id}: unexpected entry type {type(entry)}")

        meta = self._extract_metadata(entry)
        caption = self._extract_caption(meta, entry)
        relpath = self._extract_video_relpath(meta, entry)
        video_path = self._resolve_video_path(relpath)

        video_tchw = self._decode_video_tchw_uint8(video_path)
        video_tchw = self._apply_face_cut_and_crop(video_tchw, meta)
        video_tchw = self._uniform_trim_frames(video_tchw)
        video_tchw = self._resize_video_tchw_uint8(video_tchw)

        # [T, C, H, W] -> [C, T, H, W]
        video_cthw = video_tchw.permute(1, 0, 2, 3).contiguous()

        return {
            "videos": video_cthw,
            "prompts": caption,
            "sample_id": sample_id,
            "video_path": video_path,
        }

    def __iter__(self):
        json_files = sorted(glob.glob(self.index_json_pattern))
        if not json_files:
            raise FileNotFoundError(f"No OpenS2V index json files found by pattern: {self.index_json_pattern}")

        rank, world_size = 0, 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        consumer_id = rank * num_workers + worker_id
        num_consumers = world_size * num_workers

        rng = random.Random(self.seed + consumer_id)
        if self.shuffle_json_files:
            rng.shuffle(json_files)

        for json_idx, json_path in enumerate(json_files):
            if json_idx % num_consumers != consumer_id:
                continue

            count = 0
            for sample_id, entry in _iter_json_entries(json_path):
                if self.max_entries_per_json > 0 and count >= self.max_entries_per_json:
                    break
                try:
                    sample = self._build_sample(sample_id=sample_id, entry=entry)
                    count += 1
                    yield sample
                except Exception as e:
                    if not self.skip_broken_samples:
                        raise
                    log.warning(f"Skip broken OpenS2V sample {sample_id} in {json_path}: {e}")


def create_opens2v_dataloader(
    index_json_pattern,  # e.g. "/path/OpenS2V-5M/Jsons/total_part*.json"
    videos_root,  # e.g. "/path/OpenS2V-5M/Videos"
    batch_size,
    target_resolution="480p",
    target_aspect_ratio="16:9",
    max_frames=81,
    decoder="torchvision",
    use_face_cut=True,
    use_crop=True,
    num_workers=1,
    prefetch_factor=2,
    persistent_workers=True,
    max_entries_per_json=-1,
    skip_broken_samples=True,
    shuffle_json_files=True,
    seed=0,
):
    dataset = OpenS2VIterableDataset(
        index_json_pattern=index_json_pattern,
        videos_root=videos_root,
        target_resolution=target_resolution,
        target_aspect_ratio=target_aspect_ratio,
        max_frames=max_frames,
        decoder=decoder,
        use_face_cut=use_face_cut,
        use_crop=use_crop,
        max_entries_per_json=max_entries_per_json,
        skip_broken_samples=skip_broken_samples,
        shuffle_json_files=shuffle_json_files,
        seed=seed,
    )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dict_collation_fn,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)
