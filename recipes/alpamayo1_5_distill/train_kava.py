# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Training entry point for KAVA runs.

``alpamayo1_5_sft.train_hf`` hard-codes its trainer class, and KAVA needs
:class:`~alpamayo1_5_distill.trainer.KaVaTrainer` (per-term loss logging, and no
weight decay on the soft prompt).  Rather than fork a working 100-line entry point
just to change one name, this rebinds that name in the module: ``train()`` resolves
``ReasoningVLA_Trainer`` as a module global at call time, so the substitution takes
effect for the whole run and every other behaviour — hydra composition, deepspeed
dtype fix, wandb, config dump, checkpoint resume — stays byte-identical to the SFT
path.

Usage::

    torchrun --nproc_per_node 8 -m alpamayo1_5_distill.train_kava \
        --config-path pkg://alpamayo1_5_distill/configs \
        --config-name sft_stage1_kava_cosmos2b_lcdrive \
        data.train_dataset.kv_cache_root=/temp/achahe/.../teacher_kv_lcdrive
"""

import alpamayo1_5_sft.train_hf as base

from alpamayo1_5_distill.trainer import KaVaTrainer

base.ReasoningVLA_Trainer = KaVaTrainer

if __name__ == "__main__":
    base.train()
