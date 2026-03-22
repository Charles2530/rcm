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

"""Wan2.2 T2V JVP backbone entry.

Wan2.2 A14B Diffusers T2V exposes two transformer experts (``transformer`` and
``transformer_2``). In rCM training, the dual-expert routing is handled in the
training loop by switching teacher nets with a noise-phase boundary, while this
module keeps the student-side JVP-compatible backbone definition.
"""

from rcm.networks.wan2pt1_jvp import WanModel_JVP as _WanModel_JVP


class WanModel_T2V_JVP(_WanModel_JVP):
    """Wan2.2 T2V JVP student backbone.

    This class intentionally reuses the Wan2.1 JVP implementation because
    Wan2.1-14B and Wan2.2-14B transformer parameterization are shape-compatible
    in the released checkpoints. MoE-style dual-expert routing for Wan2.2 is
    implemented outside this module (teacher checkpoint 1/2 + boundary switch).
    """

    def __init__(
        self,
        *args,
        model_type: str = "t2v",
        moe_router: str = "dual_transformer",
        moe_boundary_ratio: float = 0.875,
        **kwargs,
    ):
        if model_type != "t2v":
            raise ValueError(f"WanModel_T2V_JVP only supports model_type='t2v', got: {model_type}")
        super().__init__(*args, model_type=model_type, **kwargs)
        self.moe_router = moe_router
        self.moe_boundary_ratio = moe_boundary_ratio


# Backward-compatible alias for configs that prefer WanModel_JVP naming.
WanModel_JVP = WanModel_T2V_JVP
