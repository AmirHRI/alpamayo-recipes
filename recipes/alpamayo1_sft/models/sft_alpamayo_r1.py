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

import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import einops
import torch

from alpamayo1_sft.models.sft_base_model import ReasoningVLAOutput, load_alpamayo1_vlm
from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo.common import misc
from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


def _resolve_alpamayo_snapshot(checkpoint_path: str) -> Path:
    """Return the directory that directly contains config.json.

    Handles both flat local dirs (``huggingface-cli download --local-dir``)
    and HF hub cache dirs (``models--nvidia--Alpamayo-R1-10B``).
    """
    p = Path(checkpoint_path)
    if (p / "config.json").exists():
        return p
    snapshots = p / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(snapshots.iterdir())
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(
        f"config.json not found under: {checkpoint_path}. "
        "Pass either a flat local dir or an HF hub cache directory."
    )


def _load_modules_from_checkpoint(
    model: Any,
    checkpoint_dir: Path,
    prefixes: tuple[str, ...],
    label: str = "tensors",
) -> None:
    """Load tensors whose keys start with any of *prefixes* from a safetensors checkpoint."""
    from safetensors.torch import load_file as load_safetensors_file

    state_dict: dict[str, torch.Tensor] = {}
    index_path = checkpoint_dir / "model.safetensors.index.json"
    single_path = checkpoint_dir / "model.safetensors"

    if index_path.exists():
        weight_map: dict[str, str] = json.load(index_path.open())["weight_map"]
        shard_to_keys: defaultdict[str, list[str]] = defaultdict(list)
        for key, shard_name in weight_map.items():
            if key.startswith(prefixes):
                shard_to_keys[shard_name].append(key)
        for shard_name, keys in shard_to_keys.items():
            shard_sd = load_safetensors_file(str(checkpoint_dir / shard_name), device="cpu")
            for key in keys:
                if key in shard_sd:
                    state_dict[key] = shard_sd[key]
    elif single_path.exists():
        shard_sd = load_safetensors_file(str(single_path), device="cpu")
        state_dict = {k: v for k, v in shard_sd.items() if k.startswith(prefixes)}

    if not state_dict:
        logger.warning(f"No {label} found in {checkpoint_dir}")
        return

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    logger.info(
        f"Loaded {len(state_dict)} {label} from {checkpoint_dir} "
        f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})"
    )


class TrainableAlpamayoR1(AlpamayoR1):
    def __init__(
        self,
        config: AlpamayoR1Config,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
        cotrain_vlm: bool = False,
        stop_grad_from_vlm: bool = True,
        stage1_vlm_checkpoint_path: str | None = None,
    ):
        super().__init__(config, pretrained_modules, original_vocab_size)

        self.cotrain_vlm = cotrain_vlm
        self.stop_grad_from_vlm = stop_grad_from_vlm

        # we only need the text config for the expert model
        if stage1_vlm_checkpoint_path is not None:
            self.vlm = load_alpamayo1_vlm(stage1_vlm_checkpoint_path, self.vlm)

        if not self.cotrain_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False
        # print the param count
        logger.info("Model parameter count:")
        param_count = misc.get_param_count(self)
        for key, value in param_count.items():
            logger.info(f"{key}: {value:,}")

    @classmethod
    def from_pretrained_vlm(
        cls,
        vlm_name_or_path: str,
        alpamayo_config_path: str,
        stage1_vlm_checkpoint_path: str | None = None,
        stage2_checkpoint_path: str | None = None,
        cotrain_vlm: bool = False,
        stop_grad_from_vlm: bool = True,
        expert_num_layers: int = 7,
    ) -> "TrainableAlpamayoR1":
        """Load TrainableAlpamayoR1 with a custom (e.g. 2B) VLM backbone.

        The expert transformer is built from the 2B VLM's ``text_config`` with
        ``expert_cfg`` overrides applied, giving a KV-cache-compatible architecture.
        Pretrained ``action_in_proj`` / ``action_out_proj`` / ``diffusion`` /
        ``action_space`` weights are transferred from the Alpamayo-R1-10B checkpoint
        (all architecturally compatible since they depend only on action dims and
        ``expert_cfg.hidden_size``, both identical across VLM sizes).
        The expert transformer layers are randomly initialised — the 10B expert's
        KV projections differ in shape (``num_kv_heads`` 4 vs 8) so they cannot
        be reused.

        Args:
            vlm_name_or_path: Local path to the base VLM (e.g. Cosmos-Reason2-2B).
                Used for processor / tokenizer and VLM architecture.
            alpamayo_config_path: Flat local dir or HF hub cache dir for
                Alpamayo-R1-10B.  Supplies ``AlpamayoR1Config`` and pretrained
                action / diffusion weights.
            stage1_vlm_checkpoint_path: Stage-1 HF Trainer checkpoint dir.
                Its ``vlm.*`` tensors overwrite the randomly initialised VLM.
            stage2_checkpoint_path: Stage-2 HF Trainer checkpoint dir (eval only).
                ALL tensors (``vlm.*``, ``expert.*``, ``action_*``) are loaded.
            cotrain_vlm: Keep VLM trainable during Stage 2 (default False).
            stop_grad_from_vlm: Detach VLM KV cache before passing to expert.
            expert_num_layers: Transformer layers in the expert.
                7 layers ≈ 0.44 B, matching the ~20 % VLM/expert ratio of
                the original Alpamayo-R1-10B (1.71 B expert / 8.8 B VLM).
        """
        from hydra.utils import instantiate

        # 1. Resolve Alpamayo-R1-10B snapshot and load its AlpamayoR1Config
        snapshot_dir = _resolve_alpamayo_snapshot(alpamayo_config_path)
        ar1_config = AlpamayoR1Config.from_pretrained(str(snapshot_dir))

        # 2. Override the VLM path so ReasoningVLA.__init__ builds a 2B architecture
        ar1_config.vlm_name_or_path = vlm_name_or_path

        # 3. Set expert depth (controls ~0.44B size and KV-head compatibility)
        ar1_config.expert_cfg = dict(ar1_config.expert_cfg or {})
        ar1_config.expert_cfg["num_hidden_layers"] = expert_num_layers

        # 4. Instantiate trajectory tokeniser (has pretrained binning weights)
        pretrained_modules: dict[str, Any] = {}
        if ar1_config.traj_tokenizer_cfg is not None:
            pretrained_modules["traj_tokenizer"] = instantiate(ar1_config.traj_tokenizer_cfg)

        # 5. Build model:
        #    - VLM = 2B architecture, random init
        #    - expert = expert_num_layers-layer transformer, random init
        #    - stage1_vlm_checkpoint_path handled inside __init__ → load_alpamayo1_vlm
        model = cls(
            ar1_config,
            pretrained_modules=pretrained_modules or None,
            cotrain_vlm=cotrain_vlm,
            stop_grad_from_vlm=stop_grad_from_vlm,
            stage1_vlm_checkpoint_path=stage1_vlm_checkpoint_path,
        )

        if stage2_checkpoint_path is not None:
            # Eval path: load every saved tensor from the Stage-2 checkpoint
            _load_modules_from_checkpoint(
                model,
                Path(stage2_checkpoint_path),
                ("vlm.", "expert.", "action_in_proj.", "action_out_proj.", "diffusion.", "action_space."),
                label="Stage-2 tensors",
            )
        else:
            # Training path: seed action / diffusion from pretrained Alpamayo-R1-10B
            _load_modules_from_checkpoint(
                model,
                snapshot_dir,
                ("action_in_proj.", "action_out_proj.", "diffusion.", "action_space."),
                label="action/diffusion tensors from Alpamayo-R1-10B",
            )

        return model

    def _process_traj_future_training(self, traj_data: dict[str, Any]) -> dict[str, Any]:
        """Process the trajectory future data for training."""
        ego_history_xyz = traj_data["ego_history_xyz"]
        ego_history_rot = traj_data["ego_history_rot"]
        ego_future_xyz = traj_data["ego_future_xyz"]
        ego_future_rot = traj_data["ego_future_rot"]
        action = self.action_space.traj_to_action(
            traj_history_xyz=ego_history_xyz,
            traj_history_rot=ego_history_rot,
            traj_future_xyz=ego_future_xyz,
            traj_future_rot=ego_future_rot,
        )
        action = action.reshape(-1, *self.action_space.get_action_space_dims())
        training_data: dict[str, Any] = self.diffusion.construct_training_data(action)
        return training_data

    def _process_position_ids_qwen2_5_vl(
        self, vlm_outputs: Any, batch_size: int, num_expert_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """Process the position ids for the expert model.

        Qwen 2.5 VL has a special RoPE, so we need to process the position ids
        Args:
            vlm_outputs: The outputs of the VLM model.
            batch_size: The batch size.
            num_expert_tokens: The number of expert tokens.
            device: The device.
        Returns:
            The processed position ids.
        """
        position_ids = torch.arange(num_expert_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch_size).clone()
        delta = vlm_outputs.rope_deltas + vlm_outputs.past_key_values.get_seq_length()
        position_ids += delta.to(position_ids.device)
        return position_ids

    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ReasoningVLAOutput:
        """Forward pass of the model."""
        # 1. tokenize trajectory and fuse into input_ids
        input_ids = tokenized_data.pop("input_ids")
        batch_size = input_ids.shape[0]
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        # 2. get labels
        labels = input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        # 3. vlm forward pass
        if self.cotrain_vlm:
            context = nullcontext()
        else:
            context = torch.no_grad()

        with context:
            vlm_outputs = self.vlm(
                input_ids=input_ids,
                labels=labels,
                use_cache=True,
                **tokenized_data,
            )

        future_start_token_id = self.config.traj_token_ids["future_start"]
        last_traj_future_start_idx = (input_ids == future_start_token_id).nonzero(as_tuple=False)
        last_traj_future_start_idx = last_traj_future_start_idx[-1, 1] + 1

        future_traj_data = self._process_traj_future_training(traj_data)
        # [B, n_token_per_future_traj, hidden_size]
        action_embeds = self.action_in_proj(
            future_traj_data["noisy_x"], future_traj_data["timesteps"]
        )
        # [B, n_token_per_history_traj + n_token_per_future_traj, hidden_size]
        expert_embeds = action_embeds
        # NOTE: we don't need to update the rope deltas as we assume after <traj_future_start> there
        # will be no more vision tokens.
        kv_cache = vlm_outputs.past_key_values
        # crop the kv cache to the last <traj_future_start> token
        kv_cache.crop(last_traj_future_start_idx)
        if self.stop_grad_from_vlm:
            for layer in kv_cache.layers:
                layer.keys = layer.keys.detach()
                layer.values = layer.values.detach()
        position_ids = self._process_position_ids_qwen2_5_vl(
            vlm_outputs, batch_size, expert_embeds.shape[1], expert_embeds.device
        )
        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False
        expert_outputs = self.expert(
            inputs_embeds=expert_embeds,
            position_ids=position_ids,
            past_key_values=kv_cache,
            attention_mask=None,
            use_cache=True,
            **forward_kwargs,
        )
        diffusion_out = expert_outputs.last_hidden_state[:, -action_embeds.shape[1] :]
        pred = self.action_out_proj(diffusion_out)
        pred = pred.view(-1, *self.action_space.get_action_space_dims())
        future_traj_loss = (
            self.diffusion.compute_loss_from_pred(
                training_data=future_traj_data,
                pred=pred,
            )
            # TODO: only support traj finetune for now, so no weight, add weight later when other losses added
            # * self.config.traj_loss_weight
        )
        loss = future_traj_loss
        if self.cotrain_vlm:
            loss += vlm_outputs.loss

        return ReasoningVLAOutput(
            loss=loss,
        )

    def sample_trajectories_from_data(  # type: ignore[override]
        self,
        data: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        """Sample trajectories from the data.

        Args:
            with_vlm_rollout: Whether to use VLM rollout.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        return self.sample_trajectories_from_data_with_vlm_rollout(
            data,
            *args,
            **kwargs,
        )
