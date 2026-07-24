# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from typing import Any, Self

import einops
import torch

from alpamayo1_5_sft_qwen3_5.models.sft_base_model import ReasoningVLAOutput, load_alpamayo1_vlm
from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.hybrid_cache import crop_cache, detach_cache_, prime_cache
from alpamayo_r1.models.multimodal_inputs import build_extra_vlm_kwargs
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo.common import misc
from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


def _resolve_alpamayo_snapshot(checkpoint_path: str) -> Path:
    """Return the directory that directly contains config.json.

    Handles both flat local dirs (``huggingface-cli download --local-dir``)
    and HF hub cache dirs (``models--nvidia--Alpamayo-1.5-10B``).
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

    # `strict=False` only tolerates missing/unexpected *keys* -- it still raises
    # on a shape mismatch for a key that exists in both. That happens here
    # whenever the new VLM's hidden_size differs from the checkpoint's (e.g. the
    # 10B checkpoint's action_in_proj/action_out_proj are sized for hidden_size
    # 2048, but a Qwen3.5-0.8B-backed expert's are sized for 1024) -- filter
    # those out explicitly and fall back to random init for just those tensors.
    target_state_dict = model.state_dict()
    shape_mismatches = [
        key
        for key in state_dict
        if key in target_state_dict and state_dict[key].shape != target_state_dict[key].shape
    ]
    for key in shape_mismatches:
        logger.warning(
            f"Skipping {key}: checkpoint shape {tuple(state_dict[key].shape)} != "
            f"model shape {tuple(target_state_dict[key].shape)} (falls back to random init)"
        )
        del state_dict[key]
    if not state_dict:
        logger.warning(f"All {label} were shape-incompatible with {checkpoint_dir}; skipping")
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
        if stage1_vlm_checkpoint_path is not None:
            raise ValueError(
                "stage1_vlm_checkpoint_path is only supported by "
                "TrainableAlpamayoR1.from_pretrained()"
            )

        super().__init__(config, pretrained_modules, original_vocab_size)

        self.cotrain_vlm = cotrain_vlm
        self.stop_grad_from_vlm = stop_grad_from_vlm

        self._set_vlm_trainability()
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
        """Load TrainableAlpamayoR1 with a custom (e.g. Qwen 3.5) VLM backbone.

        The expert transformer is built from the new VLM's own ``text_config``
        with only ``num_hidden_layers`` overridden -- unlike the Cosmos-Reason2
        variant of this method, ``hidden_size`` / ``intermediate_size`` /
        ``num_attention_heads`` / ``head_dim`` are intentionally left as the new
        VLM's own values, not inherited from the 10B checkpoint's ``expert_cfg``.
        The 10B's ``expert_cfg`` hardcodes ``head_dim: 128``, which matches
        Cosmos-Reason2 but not Qwen 3.5 (``head_dim: 256``, confirmed from the
        real Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-2B checkpoints) -- reusing it
        verbatim would build an expert whose full-attention layers can't
        actually consume the VLM's own KV-cache shape.

        One consequence: ``action_in_proj`` / ``action_out_proj`` / ``diffusion``
        can only be seed-loaded from the 10B checkpoint when the new VLM's own
        ``hidden_size`` happens to already equal 2048 (true for Qwen3.5-2B, not
        for Qwen3.5-0.8B, whose native hidden_size is 1024) -- when it doesn't
        match, `_load_modules_from_checkpoint`'s `strict=False` load silently
        drops those tensors and they fall back to random init, same as the
        expert's own transformer layers already do.

        Args:
            vlm_name_or_path: Local path or HF Hub ID for the Qwen 3.5 VLM
                (e.g. a downloaded ``Qwen/Qwen3.5-0.8B`` or ``Qwen/Qwen3.5-2B``).
            alpamayo_config_path: Flat local dir or HF hub cache dir for
                Alpamayo-1.5-10B.  Supplies ``AlpamayoR1Config`` (trajectory
                tokenizer, action space, diffusion, traj-token settings) and,
                when shapes allow, pretrained action / diffusion weights.
            stage1_vlm_checkpoint_path: Stage-1 HF Trainer checkpoint dir.
                Its ``vlm.*`` tensors overwrite the randomly initialised VLM.
            stage2_checkpoint_path: Stage-2 HF Trainer checkpoint dir (eval only).
                ALL tensors (``vlm.*``, ``expert.*``, ``action_*``) are loaded.
            cotrain_vlm: Keep VLM trainable during Stage 2 (default False).
            stop_grad_from_vlm: Detach VLM KV cache before passing to expert.
            expert_num_layers: Transformer layers in the expert. Pick this to
                land near the ~20% expert/VLM parameter ratio used by the
                released Alpamayo-1.5-10B and the Cosmos-Reason2-2B variant --
                measure real per-layer params from the downloaded checkpoint
                rather than assuming the 10B's "63M/layer" figure, since Qwen
                3.5's Gated-DeltaNet layers have a different parameter count
                than its full-attention layers.
        """
        from hydra.utils import instantiate

        # 1. Resolve Alpamayo-1.5-10B snapshot and load its AlpamayoR1Config
        snapshot_dir = _resolve_alpamayo_snapshot(alpamayo_config_path)
        ar1_config = AlpamayoR1Config.from_pretrained(str(snapshot_dir))

        # 2. Override the VLM path so ReasoningVLA.__init__ builds a Qwen 3.5 architecture.
        # `from_pretrained` above loaded `vocab_size`/`traj_token_start_idx`/`traj_token_ids`
        # as literal values from the 10B checkpoint's config.json -- computed for *its*
        # Cosmos-Reason2-8B tokenizer, not this VLM's. Plain attribute assignment doesn't
        # re-trigger the config's own tokenizer-dependent setup, so without recomputing
        # these explicitly, `resize_token_embeddings` below shrinks Qwen 3.5's embedding
        # table to the 10B's (smaller and differently-laid-out) vocab_size, truncating
        # Qwen 3.5's own image_token_id/vision_start_token_id right out of range --
        # confirmed via a real forward pass: this is what caused a CUDA device-side
        # assert deep in the vision encoder (`get_vision_bilinear_indices_and_weights`),
        # a red herring location since CUDA errors surface asynchronously at the next
        # sync point after the real fault (indexing embed_tokens with an out-of-range id).
        ar1_config.vlm_name_or_path = vlm_name_or_path
        ar1_config._initialize_vlm_config()

        # 3. Set expert depth only -- do NOT inherit the 10B's hidden_size /
        #    intermediate_size / num_attention_heads / head_dim overrides (see
        #    docstring above). Leaving expert_cfg to just num_hidden_layers means
        #    AlpamayoR1.__init__'s `deepcopy(self.vlm.config.text_config)` keeps
        #    every other field -- including layer_types, num_key_value_heads,
        #    and the linear_* Gated-DeltaNet dims -- consistent with the actual
        #    Qwen 3.5 VLM whose cache the expert will be attending into.
        ar1_config.expert_cfg = {"num_hidden_layers": expert_num_layers}

        # 4. Instantiate trajectory tokeniser (has pretrained binning weights)
        pretrained_modules: dict[str, Any] = {}
        if ar1_config.traj_tokenizer_cfg is not None:
            pretrained_modules["traj_tokenizer"] = instantiate(ar1_config.traj_tokenizer_cfg)

        # 5. Build model:
        #    - VLM = Qwen 3.5 architecture, random init
        #    - expert = expert_num_layers-layer transformer, random init
        model = cls(
            ar1_config,
            pretrained_modules=pretrained_modules or None,
            cotrain_vlm=cotrain_vlm,
            stop_grad_from_vlm=stop_grad_from_vlm,
        )

        # 6. Overwrite the randomly initialised VLM with the Stage-1 fine-tuned
        #    weights (single-file or sharded safetensors handled in load_alpamayo1_vlm).
        if stage1_vlm_checkpoint_path is not None:
            model.vlm = load_alpamayo1_vlm(
                stage1_vlm_checkpoint_path,
                model.vlm,
                preserve_model_device_and_dtype=True,
            )
            model._set_vlm_trainability()

        if stage2_checkpoint_path is not None:
            # Eval path: load every saved tensor from the Stage-2 checkpoint
            _load_modules_from_checkpoint(
                model,
                Path(stage2_checkpoint_path),
                ("vlm.", "expert.", "action_in_proj.", "action_out_proj.", "diffusion.", "action_space."),
                label="Stage-2 tensors",
            )
        else:
            # Training path: seed action / diffusion from pretrained Alpamayo-1.5-10B
            # where shapes allow (see docstring -- silently skipped via strict=False
            # for tiers whose hidden_size doesn't match the 10B's 2048).
            _load_modules_from_checkpoint(
                model,
                snapshot_dir,
                ("action_in_proj.", "action_out_proj.", "diffusion.", "action_space."),
                label="action/diffusion tensors from Alpamayo-1.5-10B",
            )

        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> Self:
        stage1_vlm_checkpoint_path = kwargs.pop("stage1_vlm_checkpoint_path", None)
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

        if stage1_vlm_checkpoint_path is not None:
            model.vlm = load_alpamayo1_vlm(
                stage1_vlm_checkpoint_path,
                model.vlm,
                preserve_model_device_and_dtype=True,
            )

        model._set_vlm_trainability()
        return model

    def _set_vlm_trainability(self) -> None:
        for param in self.vlm.parameters():
            param.requires_grad = self.cotrain_vlm

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

    def _process_position_ids_from_rope_deltas(
        self, vlm_outputs: Any, batch_size: int, num_expert_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """Process the position ids for the expert model from the VLM's rope_deltas.

        Originally written for Qwen 2.5/3-VL's 3D mRoPE bookkeeping; confirmed
        (Phase 0 spike) that Qwen 3.5's `Qwen3_5Model`/`Qwen3_5ForConditionalGeneration`
        expose the identical `self.model.rope_deltas` attribute and
        `rope_deltas` output field, so this logic is unchanged for that backend.

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
                # Pre-armed so `crop_cache` below can roll this cache back
                # correctly regardless of whether `self.vlm` has any
                # linear-attention (Gated-DeltaNet) layers -- see
                # `alpamayo_r1.models.hybrid_cache` and the Phase 0 spike notes.
                past_key_values=prime_cache(self.vlm.config),
                **build_extra_vlm_kwargs(self.vlm, input_ids),
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
        crop_cache(kv_cache, last_traj_future_start_idx)
        if self.stop_grad_from_vlm:
            detach_cache_(kv_cache)
        position_ids = self._process_position_ids_from_rope_deltas(
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
