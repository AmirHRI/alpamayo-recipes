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

"""On-disk store for the teacher's per-layer KV cache.

The existing single-vector cache (:mod:`alpamayo1_5_distill.data.distill_dataset`)
keeps every sample in **one flat safetensors file**, rewrites the whole file every
``save_every`` samples, and loads all of it into every dataloader worker.  That is
fine at 16 KiB/sample; a per-layer KV target is ~19 MiB/sample, three orders of
magnitude more, and none of those three properties survive the jump.

So this module stores **one file per sample**, bucketed by the first two hex chars
of the clip UUID (~150 files per bucket at LCDrive scale), and reads it back
through ``safe_open`` — mmap, one sample at a time, no whole-cache residency.
Resume becomes a file-existence check rather than a growing-file rewrite, and N
shards on N GPUs never touch each other's files.

Two tiers live side by side under one ``cache_root``:

``full/``
    The uncompressed CoT cache plus the eviction scores: everything needed to
    re-derive *any* compressed tier later without touching the 10B teacher.
``compressed_<tag>/``
    The top-``M`` eviction result actually read during training (~2.4 MiB/sample at
    ``M=16``), one directory per ``(M, lam, method)`` combination.

Plus two flat sidecars: ``index.*.json`` (provenance and per-key ``N_C``, written
per shard) and ``cot_text.*.jsonl`` (the teacher's generated reasoning as text,
append-only so an interrupted run keeps what it had).

Layout::

    <cache_root>/
      index.shard0.json
      cot_text.shard0.jsonl
      full/<bucket>/<clip>__<t0>.safetensors          # k_pre, v, imp, red, tfs_hidden
      compressed_M16_rkv0.1/<bucket>/<clip>__<t0>.safetensors   # k_pre, v, sel_idx
"""

import json
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT_FULL = "alpamayo_kava_full_v1"
FORMAT_COMPRESSED = "alpamayo_kava_compressed_v1"

#: Tensors in a ``full/`` entry. ``red`` is recomputable from ``k_pre`` and may be
#: absent; ``imp`` is not (it needs the answer-token attention) and never is.
FULL_TENSORS = ("k_pre", "v", "imp", "red", "tfs_hidden")


def compressed_tag(m: int, lam: float, method: str) -> str:
    """Directory name for one compressed tier, e.g. ``compressed_M16_rkv0.1``.

    ``lam`` is omitted for methods that ignore it, so ``crop``/``cosine``/``attn``
    tiers do not silently multiply into near-duplicate directories.
    """
    if method in ("crop", "cosine", "attn"):
        return f"compressed_M{int(m)}_{method}"
    return f"compressed_M{int(m)}_{method}{lam:g}"


def _fname(key: str) -> str:
    """Cache key -> filename. ``"<uuid>::<t0>"`` becomes ``"<uuid>__<t0>.safetensors"``.

    Colons are legal on ext4 but travel badly through rsync/scp and shell globs, so
    they are swapped for a double underscore.  The mapping is injective because
    neither UUIDs nor integer timestamps contain ``_``.
    """
    return key.replace("::", "__") + ".safetensors"


def _bucket(key: str) -> str:
    """First two chars of the clip UUID — keeps directories to ~150 entries."""
    return key[:2]


def entry_path(cache_root: str | Path, tier: str, key: str) -> Path:
    """Absolute path of one sample's file in ``tier`` (``"full"`` or a compressed tag)."""
    return Path(cache_root) / tier / _bucket(key) / _fname(key)


def has_entry(cache_root: str | Path, tier: str, key: str) -> bool:
    """Whether this sample is already cached — the whole of the resume check."""
    return entry_path(cache_root, tier, key).exists()


# ----------------------------------------------------------------------- write
def save_entry(
    cache_root: str | Path,
    tier: str,
    key: str,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write one sample atomically (temp file + rename).

    The rename matters for resume: a crash mid-write would otherwise leave a
    truncated file that :func:`has_entry` reports as done and ``safe_open`` then
    fails on, halfway through a 10 h job.

    safetensors metadata is ``str -> str`` only, so values are JSON-encoded when
    they are not already strings.
    """
    path = entry_path(cache_root, tier, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.contiguous().cpu() for k, v in tensors.items() if v is not None}
    meta = {"key": key}
    for k, v in (metadata or {}).items():
        meta[k] = v if isinstance(v, str) else json.dumps(v)
    tmp = path.with_suffix(".safetensors.tmp")
    save_file(payload, str(tmp), metadata=meta)
    tmp.rename(path)
    return path


def save_full_entry(
    cache_root: str | Path,
    key: str,
    k_pre: torch.Tensor,
    v: torch.Tensor,
    imp: torch.Tensor | None = None,
    red: torch.Tensor | None = None,
    tfs_hidden: torch.Tensor | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a ``full/`` entry.

    Args:
        k_pre, v: ``[L, H, N_C, D]`` pre-RoPE keys / values over the CoT span, bf16.
        imp: ``[L, H, N_C]`` answer-attention mass per CoT token, fp32.  Stored
            because it cannot be recomputed without re-running the teacher.
        red: ``[L, H, N_C]`` redundancy, fp32.  Recomputable from ``k_pre``, so it
            is optional; storing it makes recompression deterministic and cheap.
        tfs_hidden: ``[H_teacher]`` last-layer hidden at ``<traj_future_start>``,
            fp32 — the existing single-vector target, carried along so one cache
            run feeds both objectives.
    """
    meta = dict(metadata or {})
    meta.setdefault("format", FORMAT_FULL)
    meta["n_cot"] = str(int(k_pre.shape[-2]))
    return save_entry(
        cache_root,
        "full",
        key,
        {
            "k_pre": k_pre.to(torch.bfloat16),
            "v": v.to(torch.bfloat16),
            "imp": None if imp is None else imp.to(torch.float32),
            "red": None if red is None else red.to(torch.float32),
            "tfs_hidden": None if tfs_hidden is None else tfs_hidden.to(torch.float32),
        },
        meta,
    )


def save_compressed_entry(
    cache_root: str | Path,
    tag: str,
    key: str,
    k_pre: torch.Tensor,
    v: torch.Tensor,
    sel_idx: torch.Tensor | None = None,
    n_valid: int | None = None,
    tfs_hidden: torch.Tensor | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write one compressed tier entry.

    Args:
        k_pre, v: ``[L, H, M', D]`` selected pre-RoPE keys / values, bf16, where
            ``M' = min(M, N_C)``.  Padding up to ``M`` happens at collation, not
            here — the file records only real targets.
        sel_idx: ``[L, H, M']`` int32 source positions, kept for diagnostics (which
            CoT tokens survived, and whether the choice varies per head as it should).
        n_valid: ``M'``.  Also derivable from the shape; stored explicitly so a
            reader never has to infer it.
        tfs_hidden: the ``<traj_future_start>`` hidden, copied in from the ``full``
            tier so training reads one ~2.4 MiB file per sample instead of also
            opening the ~19 MiB full entry for a 16 KiB vector.
    """
    meta = dict(metadata or {})
    meta.setdefault("format", FORMAT_COMPRESSED)
    m_eff = int(n_valid if n_valid is not None else k_pre.shape[-2])
    meta["n_valid"] = str(m_eff)
    return save_entry(
        cache_root,
        tag,
        key,
        {
            "k_pre": k_pre.to(torch.bfloat16),
            "v": v.to(torch.bfloat16),
            "sel_idx": None if sel_idx is None else sel_idx.to(torch.int32),
            "tfs_hidden": None if tfs_hidden is None else tfs_hidden.to(torch.float32),
        },
        meta,
    )


class CotTextLog:
    """Append-only JSONL log of the teacher's reasoning text.

    One file per shard, flushed per record: an interrupted 10 h run keeps every CoT
    it produced.  This is also the first place to look when the cache comes out
    empty — a ``traj_future``-last prompt pre-fills ``<|traj_future_start|>`` and
    the teacher emits **no** CoT at all, which shows up here as empty strings long
    before any loss curve would reveal it.
    """

    def __init__(self, cache_root: str | Path, shard: int = 0) -> None:
        self.path = Path(cache_root) / f"cot_text.shard{int(shard)}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def append(self, key: str, text: str, n_cot: int, **extra: Any) -> None:
        if self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8")
        record = {"key": key, "n_cot": int(n_cot), "cot_text": text, **extra}
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "CotTextLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def write_index(
    cache_root: str | Path,
    entries: dict[str, int],
    metadata: dict[str, Any],
    shard: int = 0,
) -> Path:
    """Write ``index.shard<N>.json``: ``{key: N_C}`` plus run provenance."""
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"index.shard{int(shard)}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "n_cot": entries}, f, indent=2)
    return path


def read_index(cache_root: str | Path) -> dict[str, Any]:
    """Merge every ``index.shard*.json`` into one ``{"metadata", "n_cot"}`` dict.

    Returns empty dicts when no index exists — the readers below only need the
    files themselves, so a missing or partial index degrades QA output rather than
    breaking training.
    """
    root = Path(cache_root)
    merged: dict[str, int] = {}
    metadata: dict[str, Any] = {}
    for path in sorted(root.glob("index.shard*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        merged.update(payload.get("n_cot", {}))
        metadata.update(payload.get("metadata", {}))
    return {"metadata": metadata, "n_cot": merged}


# ------------------------------------------------------------------------ read
def load_entry(
    cache_root: str | Path,
    tier: str,
    key: str,
    names: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """mmap-read one sample, optionally only some tensors.

    ``names`` skips what a caller does not need — e.g. a ``cosine``-only
    recompression never touches ``imp``, and training never touches ``sel_idx``.
    """
    path = entry_path(cache_root, tier, key)
    if not path.exists():
        raise KeyError(
            f"no '{tier}' cache entry for key {key!r} at {path}. Re-run "
            "scripts/generate_teacher_kv.py (or compress_teacher_kv.py) over the "
            "same dataset config."
        )
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        available = set(f.keys())
        for name in names if names is not None else sorted(available):
            if name in available:
                out[name] = f.get_tensor(name)
    return out


def load_entry_metadata(cache_root: str | Path, tier: str, key: str) -> dict[str, str]:
    """Read just the string metadata header of one entry (no tensor pages touched)."""
    path = entry_path(cache_root, tier, key)
    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})


def iter_keys(cache_root: str | Path, tier: str) -> Iterator[str]:
    """Yield every cached key in ``tier`` by walking the bucket directories."""
    root = Path(cache_root) / tier
    if not root.exists():
        return
    for path in sorted(root.glob("*/*.safetensors")):
        yield path.stem.replace("__", "::")


def tier_stats(cache_root: str | Path, tier: str) -> dict[str, Any]:
    """Count entries and bytes in a tier — for the pilot's disk extrapolation."""
    root = Path(cache_root) / tier
    if not root.exists():
        return {"tier": tier, "n_entries": 0, "bytes": 0, "mb_per_entry": 0.0}
    sizes = [p.stat().st_size for p in root.glob("*/*.safetensors")]
    total = sum(sizes)
    return {
        "tier": tier,
        "n_entries": len(sizes),
        "bytes": total,
        "mb_per_entry": (total / len(sizes) / 1024**2) if sizes else 0.0,
    }
