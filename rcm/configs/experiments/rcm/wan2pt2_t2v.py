# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict
from rcm.utils.timestep_utils import LogNormal, UniformShift


def build_debug_run(job):
    return dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=dict(
            max_iter=25,
            logging_iter=2,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=6,
                    num_samples=2,
                ),
                every_n_sample_ema=dict(
                    every_n=6,
                    num_samples=2,
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=10,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        model=dict(
            config=dict(
                tangent_warmup=0,
            )
        ),
    )


WAN2PT2_A14B_RES480P_T2V_SCM: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_distill_rcm"},
            {"override /net": "wan2pt2_a14b_t2v_jvp"},
            {"override /net_teacher": "wan2pt2_a14b_t2v"},
            {"override /net_fake_score": None},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_distill",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="rCM_Wan",
            name="wan2pt2_a14b_res480p_t2v_scm",
        ),
        optimizer=dict(
            lr=1e-6,
            weight_decay=0.01,
            betas=(0.0, 0.999),
        ),
        model=dict(
            config=dict(
                loss_scale=100.0,
                loss_scale_dmd=0.0,
                fsdp_shard_size=8,
                resolution="480p",
                p_G=L(LogNormal)(p_mean=1.5, p_std=1.6),
                p_D=L(UniformShift)(shift=5.0),
                max_simulation_steps_fake=4,
                state_t=21,
                sigma_max=200,
                grad_clip=False,
                rectified_flow_t_scaling_factor=1000.0,
                student_update_freq=10,
                tokenizer=dict(vae_pth="./model/Wan2.1_VAE.pth"),
                text_encoder_path="./model/models_t5_umt5-xxl-enc-bf16.pth",
                neg_embed_path="./model/umT5_wan_negative_emb.pt",
                teacher_ckpt="./model/Wan-AI/Wan2.2-T2V-A14B-Diffusers-rcm/Wan2.2-T2V-A14B-transformer-rcm.pth",
                teacher_ckpt_2="./model/Wan-AI/Wan2.2-T2V-A14B-Diffusers-rcm/Wan2.2-T2V-A14B-transformer_2-rcm.pth",
                teacher_boundary_ratio=0.875,
                teacher_guidance=5.0,
                tangent_warmup=0,
                precision="bfloat16",
                net=dict(
                    sac_config=dict(
                        mode="mm_only",
                    ),
                ),
            )
        ),
        checkpoint=dict(
            save_iter=250,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=250,
                    num_samples=3,
                    run_at_start=True,
                ),
                every_n_sample_ema=dict(
                    every_n=250,
                    num_samples=3,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        dataloader_train=dict(
            tar_path_pattern="./datasets/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

WAN2PT2_A14B_RES480P_T2V_RCM: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{WAN2PT2_A14B_RES480P_T2V_SCM['job']['name']}",
            {"override /net_fake_score": "wan2pt2_a14b_t2v"},
            {"override /optimizer_fake_score": "fusedadamw"},
            "_self_",
        ],
        job=dict(
            group="rCM_Wan",
            name="wan2pt2_a14b_res480p_t2v_rcm",
        ),
        model=dict(
            config=dict(
                loss_scale_dmd=1.0,
                tangent_warmup=1000,
                optimizer_fake_score=dict(
                    lr=1e-7,
                    weight_decay=0.01,
                    betas=(0.0, 0.999),
                ),
            )
        ),
    ),
    flags={"allow_objects": True},
)

WAN2PT2_A14B_RES480P_T2V_SCM_OPENS2V: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{WAN2PT2_A14B_RES480P_T2V_SCM['job']['name']}",
            {"override /data_train": "opens2v"},
            "_self_",
        ],
        job=dict(
            group="rCM_Wan",
            name="wan2pt2_a14b_res480p_t2v_scm_opens2v",
        ),
        dataloader_train=dict(
            index_json_pattern="./datasets/OpenS2V-5M/Jsons/total_part*.json",
            videos_root="./datasets/OpenS2V-5M/Videos",
            batch_size=1,
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
        ),
    ),
    flags={"allow_objects": True},
)

WAN2PT2_A14B_RES480P_T2V_RCM_OPENS2V: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{WAN2PT2_A14B_RES480P_T2V_RCM['job']['name']}",
            {"override /data_train": "opens2v"},
            "_self_",
        ],
        job=dict(
            group="rCM_Wan",
            name="wan2pt2_a14b_res480p_t2v_rcm_opens2v",
        ),
        dataloader_train=dict(
            index_json_pattern="./datasets/OpenS2V-5M/Jsons/total_part*.json",
            videos_root="./datasets/OpenS2V-5M/Videos",
            batch_size=1,
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
        ),
    ),
    flags={"allow_objects": True},
)


cs = ConfigStore.instance()

job_list = [
    WAN2PT2_A14B_RES480P_T2V_SCM,
    WAN2PT2_A14B_RES480P_T2V_RCM,
    WAN2PT2_A14B_RES480P_T2V_SCM_OPENS2V,
    WAN2PT2_A14B_RES480P_T2V_RCM_OPENS2V,
]

for job in job_list:
    cs.store(group="experiment", package="_global_", name=job["job"]["name"], node=job)
    cs.store(group="experiment", package="_global_", name=job["job"]["name"] + "_debug", node=build_debug_run(job))
