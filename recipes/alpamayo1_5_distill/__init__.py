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

"""Latent-reasoning knowledge-distillation recipe for Alpamayo-1.5.

Distils the released Alpamayo-1.5-10B teacher into a Cosmos-Reason2-2B student
by matching the student's hidden state at ``<traj_future_start>`` (the KV the
action expert consumes) to the teacher's *CoT-conditioned* representation, so
the student needs no autoregressive reasoning tokens at inference.

This recipe reuses ``alpamayo1_5_sft``'s model classes, trainer, and shared
config groups; only the distillation-specific model subclass, dataset wrappers,
offline teacher-feature cache script, and configs live here.
"""
