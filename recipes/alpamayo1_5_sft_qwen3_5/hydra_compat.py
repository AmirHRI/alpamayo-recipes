# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workaround for a transformers>=5.2 / hydra-core<=1.3 incompatibility.

`transformers.PreTrainedConfig` is a `@dataclass` as of transformers v5 (it
wasn't in 4.x). Every call site that does
`hyu.instantiate(cfg_node, model_config=model.config)` -- both in this recipe
(train_hf.py, evaluate_hf.py, profile_qwen3_5_inference.py) and inside the
shared `alpamayo.data.pai.PAIDataset.__init__`'s own internal
`instantiate(vla_preprocess_args, model_config=model_config)` -- passes a
`PreTrainedConfig` instance as an extra kwarg. Hydra's `instantiate()` merges
extra kwargs with the config node through OmegaConf, which detects a
dataclass-typed value and tries to introspect it via
`typing.get_type_hints()`. That fails with `NameError: name 'torch' is not
defined`, because some transformers-internal config field's type annotation is
only imported under `TYPE_CHECKING` (fine for static type checkers, invisible
to a real runtime `get_type_hints()` call) -- independent of which VLM
backend/config subclass is involved, and reproduces under every `_convert_`
mode, so it isn't something a local config change can route around.

Rather than patching the shared `alpamayo.data.pai` module (used by every
recipe, most of which are still on transformers 4.x where this never
triggers), wrap `model.config` once at this recipe's own call sites: the proxy
isn't a dataclass, so OmegaConf treats it as an opaque object instead of
trying to introspect it, and the wrapping is transparent to every consumer
here (`get_preprocess_data_fn_from_model_config` /
`collate_fn_from_model_config` / `PAIDataset.__init__`), which all only ever
do plain attribute access on `model_config`.
"""

import dataclasses
from typing import Any


class _OpaqueConfigProxy:
    def __init__(self, target: Any) -> None:
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_target"), name)


def opaque_model_config(config: Any) -> Any:
    """Wrap `config` so it isn't detected as a dataclass by OmegaConf.

    No-op (returns `config` unchanged) when it isn't a dataclass in the first
    place -- i.e. always, under transformers<5.2 -- so this has zero effect on
    any other recipe or on this one if the transformers pin ever moves back.
    """
    if not dataclasses.is_dataclass(config):
        return config
    return _OpaqueConfigProxy(config)
