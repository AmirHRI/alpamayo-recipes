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

"""Entry point for the Qwen3-VL-4B KD runs.

Identical to ``alpamayo1_5_sft.train_hf`` except for the Trainer, which logs the loss
terms separately and probes each one's share of the backbone gradient.  With three terms
that is not a nicety: a term can be finite, decreasing, and steering nothing.

Rebinding the module global rather than duplicating ``train()`` keeps hydra composition,
the deepspeed dtype fix, the config dump and resume byte-identical to the sft recipe.
``train()`` resolves ``ReasoningVLA_Trainer`` at call time, which is what makes this work.
"""

import alpamayo1_5_sft.train_hf as base

from alpamayo1_5_distill.trainer import KaVaTrainer

base.ReasoningVLA_Trainer = KaVaTrainer

if __name__ == "__main__":
    base.train()
