# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from imaginaire.utils.io import save_image_or_video
from imaginaire.lazy_config import LazyCall as L, LazyDict, instantiate
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.utils.model_utils import init_weights_on_device, load_state_dict
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.networks.wan2pt1 import WanModel

torch._dynamo.config.suppress_errors = True

WAN2PT1_1PT3B_T2V: LazyDict = L(WanModel)(
    dim=1536,
    eps=1e-06,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=16,
    model_type="t2v",
    num_heads=12,
    num_layers=30,
    out_dim=16,
    text_len=512,
)

WAN2PT1_14B_T2V: LazyDict = L(WanModel)(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=16,
    model_type="t2v",
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
)

dit_configs = {"1.3B": WAN2PT1_1PT3B_T2V, "14B": WAN2PT1_14B_T2V}

tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

_DEFAULT_PROMPT = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rCM inference script for Wan2.1 T2V")
    parser.add_argument("--model_size", choices=["1.3B", "14B"], default="1.3B", help="Size of the model to use")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4, help="1~4 for timestep-distilled inference")
    parser.add_argument("--sigma_max", type=float, default=80, help="Initial sigma for rCM")
    parser.add_argument("--dit_path", type=str, default="", help="Custom path to the DiT model checkpoint for distilled models.")
    parser.add_argument("--vae_path", type=str, default="model/Wan2.1-T2V-distill/Wan2.1_VAE.pth", help="Path to the Wan2.1 VAE.")
    parser.add_argument(
        "--text_encoder_path", type=str, default="model/Wan2.1-T2V-distill/models_t5_umt5-xxl-enc-bf16.pth", help="Path to the umT5 text encoder."
    )
    parser.add_argument("--num_frames", type=int, default=77, help="Number of frames to generate")
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT, help="Text prompt for video generation")
    parser.add_argument("--resolution", default="480p", type=str, help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str, help="Aspect ratio of the generated output (width:height)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path", type=str, default="output/generated_video.mp4", help="Path to save the generated video (include file extension)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    with init_weights_on_device():
        net = instantiate(dit_configs[args.model_size]).eval()  # inference

    state_dict = load_state_dict(args.dit_path)
    prefix_to_load = "net."
    # drop net. prefix
    state_dict_dit_compatible = dict()
    for k, v in state_dict.items():
        if k.startswith(prefix_to_load):
            state_dict_dit_compatible[k[len(prefix_to_load) :]] = v
        else:
            state_dict_dit_compatible[k] = v
    net.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
    del state_dict, state_dict_dit_compatible
    log.success(f"Successfully loaded DiT from {args.dit_path}")

    net.to(**tensor_kwargs).cpu()
    torch.cuda.empty_cache()

    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    log.info(f"Computing embedding for prompt: {args.prompt}")
    text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt).to(dtype=torch.bfloat16).cuda()
    clear_umt5_memory()

    log.info(f"Generating with prompt: {args.prompt}")
    condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples)}

    to_show = []

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    # mid_t = [1.3, 1.0, 0.6][: args.num_steps - 1]
    # For better visual quality
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]

    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )

    # Convert TrigFlow timesteps to RectifiedFlow
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    # Sampling steps
    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1
    net.cuda()
    for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps)):
        with torch.no_grad():
            v_pred = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs), **condition).to(
                torch.float64
            )
            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                *x.shape,
                dtype=torch.float32,
                device=tensor_kwargs["device"],
                generator=generator,
            )
    samples = x.float()

    video = tokenizer.decode(samples)

    to_show.append(video.float().cpu())

    to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

    save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), args.save_path, fps=16)
