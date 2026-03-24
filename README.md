<h1 align="center"> rCM: Score-Regularized Continuous-Time Consistency Model <br>🚀SOTA JVP-Based Diffusion Distillation & Few-Step Video Generation & Scaling Up sCM/MeanFlow </h1>
<div align="center">
  <div>
  <p align="center" style="font-size: larger;">
    <strong>ICLR 2026</strong>
  </p>
  </div>
  <a href='https://arxiv.org/abs/2510.08431'><img src='https://img.shields.io/badge/Paper%20(arXiv)-2510.08431-red?logo=arxiv'></a>  &nbsp;
  <a href='https://research.nvidia.com/labs/dir/rcm'><img src='https://img.shields.io/badge/Website-green?logo=homepage&logoColor=white'></a> &nbsp;
</div>

## Overview

rCM is the first work that:
- Scales up **continuous-time consistency distillation (e.g., sCM/MeanFlow)** to 10B+ parameter video diffusion models.
- Provides open-sourced **FlashAttention-2 Jacobian-vector product (JVP) kernel** with support for parallelisms like FSDP/CP.
- Identifies the quality bottleneck of sCM and overcomes it via a **forward–reverse divergence joint distillation** framework.
- Delivers models that generate videos with both **high quality and strong diversity in only 2~4 steps**.

#### Comparison with Other Diffusion Distillation Methods on Wan2.1 T2V 1.3B (4-step)

| teacher | DMD2 | SiD |
| :---: | :---: | :---: |
| <video src="https://github.com/user-attachments/assets/cdcd9fff-5ae9-4ba9-8864-4e1d733c1ce1" alt="teacher" controls></video> | <video src="https://github.com/user-attachments/assets/3f1ad494-9f13-4b2f-bf3e-b99ef98dbae4" alt="DMD2" controls></video> | <video src="https://github.com/user-attachments/assets/2cbd9f62-3d2a-4170-ad9a-a7f59d534ad3" alt="SiD" controls></video> |
|**sCM**|**rCM (Ours)**||
| <video src="https://github.com/user-attachments/assets/50693577-9a32-4b98-86ad-d4e1be4affdc" alt="sCM" controls></video> | <video src="https://github.com/user-attachments/assets/3da35a11-8ce6-4232-9aa2-6b3bc8b7cabf" alt="rCM" controls></video> | |

rCM achieves both **high quality** and **strong diversity**.

#### Performance under Fewer (1~2) Steps

| 1-step | 2-step | 4-step |
| --- | --- | --- |
| <video src="https://github.com/user-attachments/assets/fffab30d-de3f-4b86-b3b6-54208761d18b" alt="1-step" controls></video> | <video src="https://github.com/user-attachments/assets/e5477835-861f-4333-a99e-040b99186de5" alt="2-step" controls></video> | <video src="https://github.com/user-attachments/assets/8c39b50e-72df-411b-8c8e-ef69a5d3431f" alt="4-step" controls></video> |

#### 5 Random Videos with Distilled Wan2.1 T2V 14B (4-step)

<video src="https://github.com/user-attachments/assets/b1e3b786-134b-429d-b859-840646502c9b" controls></video>

## Environment Setup
Our training and inference are based on native PyTorch, completely free from `accelerate` and `diffusers`.

```bash
conda create -n rcm python==3.12.12
conda activate rcm
conda install cmake ninja
conda install -c nvidia cuda-nvcc cuda-toolkit
# depending on your cuda version
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
# misc
pip install megatron-core hydra-core loguru attrs fvcore nvidia-ml-py imageio[ffmpeg] pandas wandb psutil ftfy regex transformers webdataset
# transformer_engine
pip install --no-build-isolation transformer_engine[pytorch]
# flash_attn
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.7.4.post1
MAX_JOBS=4 python setup.py install
```

## Inference

Below is an example inference script for running rCM on T2V:

```bash
# Basic usage:
#   PYTHONPATH=. python rcm/inference/wan2pt1_t2v_rcm_infer.py [arguments]

# Arguments:
# --model_size         Model size: "1.3B" or "14B" (default: 1.3B)
# --num_samples        Number of videos to generate (default: 1)
# --num_steps          Sampling steps, 1–4 (default: 4)
# --sigma_max          Initial sigma for rCM (default: 80); larger choices (e.g., 1600) reduce diversity but may enhance quality
# --dit_path           Path to the distilled DiT model checkpoint (REQUIRED for inference)
# --vae_path           Path to Wan2.1 VAE (default: model/Wan2.1-T2V-distill/Wan2.1_VAE.pth)
# --text_encoder_path  Path to umT5 text encoder (default: model/Wan2.1-T2V-distill/models_t5_umt5-xxl-enc-bf16.pth)
# --prompt             Text prompt for video generation (default: A stylish woman walks down a Tokyo street...)
# --resolution         Output resolution, e.g. "480p", "720p" (default: 480p)
# --aspect_ratio       Aspect ratio in W:H format (default: 16:9)
# --seed               Random seed for reproducibility (default: 0)
# --save_path          Output file path including extension (default: output/generated_video.mp4)


# Example
PYTHONPATH=.  python rcm/inference/wan2pt1_t2v_rcm_infer.py \
    --dit_path model/Wan2.1-T2V-distill/rCM_Wan2.1_T2V_1.3B_480p.pt \
    --num_samples 5 \
    --prompt "A cinematic shot of a snowy mountain at sunrise"
```

See [Wan examples](Wan.md) for additional usage and I2V examples.

## Training
In this repo, we provide training code based on Wan2.1 and its synthetic data.

**Advanced training infrastructure, including FSDP2, Ulysses Context Parallel (CP), and Selective Activation Checkpointing (SAC), is supported**. When enabling CP, ensure that the number of GPUs is divisible by the chosen CP size, and note that the effective batch size is reduced by a factor of the CP size. 

Our training code also support:
- **Pure DMD distillation** by disabling the sCM loss (setting `model.config.loss_scale=0`), and optionally fixing the backward simulation timesteps to predetermined values (setting `model.config.dmd_fix_timesteps=True`).
- **Pure sCM distillation** by setting `model.config.net_fake_score=null` or `model.config.loss_scale_dmd=0`.

#### Key Components
- FlashAttention-2 JVP kernel: `rcm/utils/flash_attention_jvp_triton.py`
- JVP-adapted Wan2.1 student network: `rcm/networks/wan2pt1_jvp.py`
- Wan2.2 T2V JVP student entry: `rcm/networks/wan2pt2_t2v_jvp.py`
- Training loop: `rcm/models/t2v_model_distill_rcm.py`

#### Wan2.2 T2V A14B Prototype (Diffusers -> rCM)
This repo now includes a research-prototype adaptation path for `Wan-AI/Wan2.2-T2V-A14B-Diffusers`:

> Current status note:
> - Teacher side: dual experts (`transformer` / `transformer_2`) are supported with boundary-based routing.
> - Student side: current JVP path is a dense-student approximation (`wan2pt2_t2v_jvp.py` reuses Wan2.1 JVP implementation).
> - This should be treated as a research prototype baseline, not a finalized Wan2.2 MoE student implementation.

1) Convert Diffusers transformer experts (`transformer`, `transformer_2`) to rCM key format:
```bash
PYTHONPATH=. python scripts/convert_wan22_diffusers_to_rcm.py \
    --model_root ./model/Wan2.2-T2V-A14B-Diffusers \
    --output_dir ./model/Wan2.2-T2V-A14B-Diffusers-rcm
```
This conversion script outputs `.pth` teacher checkpoints (`net.*` keys), and the training loader can consume them directly (no mandatory `.dcp` conversion for Wan2.2).

2) Start with pure sCM distillation config:
```bash
torchrun --nproc_per_node=8 \
    -m scripts.train --config=rcm/configs/registry_distill.py -- \
    experiment=wan2pt2_a14b_res480p_t2v_scm \
    dataloader_train.tar_path_pattern=./datasets/shard*.tar
```

3) For joint rCM+DMD, switch to:
```bash
experiment=wan2pt2_a14b_res480p_t2v_rcm
```

You can also use the unified launcher and tune initialization/routing explicitly:
```bash
STAGE=scm DATA_BACKEND=webdataset \
TEACHER_INIT_STRATEGY=average \
TEACHER_INIT_LOW_NOISE_WEIGHT=0.5 \
TEACHER_INIT_MODULE_AWARE=false \
TEACHER_INIT_LOW_NOISE_WEIGHT_EMBED=0.5 \
TEACHER_INIT_LOW_NOISE_WEIGHT_EARLY=0.5 \
TEACHER_INIT_LOW_NOISE_WEIGHT_LATE=0.5 \
TEACHER_INIT_LOW_NOISE_WEIGHT_HEAD=0.5 \
TEACHER_BOUNDARY_RATIO=0.875 \
bash scripts/train_wan22.sh
```
`TEACHER_INIT_STRATEGY` supports `teacher_1|teacher_2|average`.
`TEACHER_INIT_LOW_NOISE_WEIGHT` applies when `TEACHER_INIT_STRATEGY=average`.
When `TEACHER_INIT_MODULE_AWARE=true`, per-module low-noise blend weights are used for embed/early/late/head groups.
`TEACHER_BOUNDARY_RATIO` controls the high-noise (`transformer`) to low-noise (`transformer_2`) switch point.
Current script/config defaults are neutral heuristics for baseline safety, not validated global optima.

Before training, run a parity check to verify converted teachers:
```bash
PYTHONPATH=. python scripts/check_wan22_teacher_parity.py \
    --model_root ./model/Wan2.2-T2V-A14B-Diffusers \
    --converted_root ./model/Wan2.2-T2V-A14B-Diffusers-rcm \
    --dtypes float32,bfloat16 \
    --seeds 0,1,2,3,4,5,6,7 \
    --timesteps 0,50,250,500,750,999 \
    --latent_shapes 21x60x106,21x48x84 \
    --save_json ./outputs/wan22_teacher_parity.json \
    --strict
```

To inspect boundary-based dual-teacher routing behavior, run:
```bash
PYTHONPATH=. python scripts/check_wan22_teacher_routing.py \
    --boundary_ratios 0.75,0.8,0.875,0.9 \
    --save_json ./outputs/wan22_teacher_routing.json
```

For pipeline-level teacher smoke (`denoise(net_type="teacher")`, boundary-near continuity and route composition), run:
```bash
PYTHONPATH=. python scripts/check_wan22_teacher_pipeline_smoke.py \
    --model_root ./model/Wan2.2-T2V-A14B-Diffusers \
    --converted_root ./model/Wan2.2-T2V-A14B-Diffusers-rcm \
    --teacher_boundary_ratio 0.875 \
    --dtype float32 \
    --save_json ./outputs/wan22_teacher_pipeline_smoke.json \
    --strict
```

For training-state preflight (real `T2VDistillModel_rCM` instantiation + init consistency + `student_F_withT` finite differences + teacher pipeline consistency), run:
```bash
PYTHONPATH=. python scripts/check_wan22_training_state_preflight.py \
    --config rcm/configs/registry_distill.py \
    --experiment wan2pt2_a14b_res480p_t2v_scm \
    --teacher_ckpt ./model/Wan2.2-T2V-A14B-Diffusers-rcm/Wan2.2-T2V-A14B-transformer-rcm.pth \
    --teacher_ckpt_2 ./model/Wan2.2-T2V-A14B-Diffusers-rcm/Wan2.2-T2V-A14B-transformer_2-rcm.pth \
    --precision float32 \
    --batch_size 1 \
    --save_json ./outputs/wan22_training_state_preflight.json \
    --strict
```

For training-path JVP sanity (`student_F_withT`), run:
```bash
PYTHONPATH=. python scripts/check_wan22_student_fwitht_sanity.py \
    --preset small \
    --dtype float32 \
    --save_json ./outputs/wan22_student_fwitht_sanity.json \
    --strict
```

For local operator-level JVP sanity (small dense proxy), run:
```bash
PYTHONPATH=. python scripts/check_wan22_jvp_sanity.py \
    --dtype float32 \
    --save_json ./outputs/wan22_jvp_sanity.json \
    --strict
```

You can also gate training with preflight checks directly from launcher scripts:
```bash
WAN22_PREFLIGHT=true \
WAN22_PREFLIGHT_PARITY=true \
WAN22_PREFLIGHT_ROUTING=true \
WAN22_PREFLIGHT_TRAINING_STATE=true \
WAN22_PREFLIGHT_DTYPE=float32 \
WAN22_PREFLIGHT_BATCH_SIZE=1 \
bash scripts/train_wan22.sh
```
`WAN22_PREFLIGHT=true` enables checks and fails fast when strict checks fail.
`WAN22_PREFLIGHT_PARITY=true` runs a reduced strict parity regression before training.
`WAN22_PREFLIGHT_TRAINING_STATE=true` runs strict training-state preflight (model init + `student_F_withT` + teacher pipeline).

For a minimal short-run ablation matrix on defaults/boundary, run:
```bash
MAX_ITER=300 STAGE=scm DATA_BACKEND=webdataset \
bash scripts/run_wan22_short_ablation.sh
```

#### OpenS2V-5M Raw Video Training
`OpenS2V-5M` provides metadata in `Jsons/total_part*.json` with fields such as:
- `metadata.path` (relative video path)
- `metadata.cap` (caption list)
- `metadata.crop` and `metadata.face_cut` (recommended crop/cut ranges)

This repo now includes an OpenS2V dataloader (`data_train=opens2v`) that reads raw videos from:
- `.../Jsons/total_part*.json`
- `.../Videos`

and computes `t5_text_embeddings` online from `prompts` when missing.
If your `Videos` are still `*.tar.split*`, first recover and extract them (as described in the OpenS2V dataset card).

For 8-GPU training, use:
```bash
bash scripts/train_wan22_opens2v_8xh800.sh
```

#### Checkpoints Downloading
Download the Wan2.1 teacher checkpoints in `.pth` format and VAE/text encoder to `model/Wan2.1-T2V-distill`:

```bash
# make sure git lfs is installed
git clone https://huggingface.co/worstcoder/Wan model/Wan2.1-T2V-distill
```

Our training checkpoint save/resume path is based on [Distributed Checkpoint (DCP)](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html).
For teacher model loading, the code supports both regular checkpoints (`.pth` / `.safetensors`) and DCP directories.

For Wan2.1 teacher checkpoints in this README example, convert `.pth` to `.dcp` first:

```bash
python -m torch.distributed.checkpoint.format_utils torch_to_dcp model/Wan2.1-T2V-distill/Wan2.1-T2V-1.3B.pth model/Wan2.1-T2V-distill/Wan2.1-T2V-1.3B.dcp
```

For Wan2.2 prototype training, the converted experts from `scripts/convert_wan22_diffusers_to_rcm.py` are already `.pth` and can be used directly as `model.config.teacher_ckpt` / `model.config.teacher_ckpt_2`.

After training, the saved `.dcp` checkpoints can be converted to `.pth` using the script `scripts/dcp_to_pth.py`.

#### Dataset Downloading

We provide Wan2.1-14B-synthesized dataset with prompts from [https://huggingface.co/gdhe17/Self-Forcing/resolve/main/vidprom_filtered_extended.txt](https://huggingface.co/gdhe17/Self-Forcing/resolve/main/vidprom_filtered_extended.txt). Download to `assets/datasets` using:

```bash
# make sure git lfs is installed
git clone https://huggingface.co/datasets/worstcoder/Wan_datasets assets/datasets
```

#### Start Training
Single-node training example:

```bash
WORKDIR="/path/to/rcm"
cd $WORKDIR
export PYTHONPATH=.

# the "IMAGINAIRE_OUTPUT_ROOT" environment variable is the path to save experiment output files
export IMAGINAIRE_OUTPUT_ROOT=${WORKDIR}/outputs
CHECKPOINT_ROOT=${WORKDIR}/model/Wan2.1-T2V-distill
DATASET_ROOT=${WORKDIR}/assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K

# your Wandb information
export WANDB_API_KEY=xxx
export WANDB_ENTITY=xxx

registry=registry_distill
experiment=wan2pt1_1pt3B_res480p_t2v_rCM

torchrun --nproc_per_node=8 \
    -m scripts.train --config=rcm/configs/${registry}.py -- experiment=${experiment} \
        model.config.teacher_ckpt=${CHECKPOINT_ROOT}/Wan2.1-T2V-1.3B.dcp \
        model.config.tokenizer.vae_pth=${CHECKPOINT_ROOT}/Wan2.1_VAE.pth \
        model.config.text_encoder_path=${CHECKPOINT_ROOT}/models_t5_umt5-xxl-enc-bf16.pth \
        model.config.neg_embed_path=${CHECKPOINT_ROOT}/umT5_wan_negative_emb.pt \
        dataloader_train.tar_path_pattern=${DATASET_ROOT}/shard*.tar
```

Please refer to `rcm/configs/experiments/rcm/wan2pt1_t2v.py` for the 14B config or perform modifications as needed.

## Future Directions

There are promising directions to explore based on rCM. For example:
- The forward–reverse divergence joint distillation framework of rCM could be extended to **autoregressive video diffusion** by leveraging a *causal teacher with teacher forcing* to complement self-forcing. 
- Few-step distilled models lag behind the teacher in aspects such as physical consistency; this can potentially be improved via reinforcement learning.

## Acknowledgement
We thank the [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2) and [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) project for providing the awesome open-source video diffusion training codebase.

## Citation
```
@article{zheng2025rcm,
  title={Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency},
  author={Zheng, Kaiwen and Wang, Yuji and Ma, Qianli and Chen, Huayu and Zhang, Jintao and Balaji, Yogesh and Chen, Jianfei and Liu, Ming-Yu and Zhu, Jun and Zhang, Qinsheng},
  journal={arXiv preprint arXiv:2510.08431},
  year={2025}
}
```
