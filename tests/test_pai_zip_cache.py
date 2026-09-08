# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface


def _cache_interface(dataset_root, cache_root, max_bytes):
    interface = PhysicalAIAVDatasetLocalInterface.__new__(
        PhysicalAIAVDatasetLocalInterface
    )
    interface.local_dir = str(dataset_root)
    interface.zip_cache_dir = str(cache_root)
    interface._zip_cache_max_bytes = max_bytes
    cache_root.mkdir(parents=True)
    return interface


def test_open_chunk_file_copies_once_and_reuses_cache(tmp_path):
    dataset_root = tmp_path / "dataset"
    source = dataset_root / "camera" / "front" / "front.chunk_0001.zip"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"first")
    cache_root = tmp_path / "cache"
    interface = _cache_interface(dataset_root, cache_root, max_bytes=1024)

    with interface._open_chunk_file(str(source)) as cached_file:
        assert cached_file.read() == b"first"

    cached_path = cache_root / source.relative_to(dataset_root)
    assert cached_path.read_bytes() == b"first"

    # Equal size represents an immutable source shard; the cached copy must be reused.
    source.write_bytes(b"later")
    with interface._open_chunk_file(str(source)) as cached_file:
        assert cached_file.read() == b"first"


def test_open_chunk_file_evicts_oldest_closed_path(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    first = dataset_root / "first.zip"
    second = dataset_root / "second.zip"
    first.write_bytes(b"a" * 8)
    second.write_bytes(b"b" * 8)
    cache_root = tmp_path / "cache"
    interface = _cache_interface(dataset_root, cache_root, max_bytes=12)

    with interface._open_chunk_file(str(first)) as cached_file:
        assert cached_file.read() == b"a" * 8
    cached_first = cache_root / "first.zip"
    os.utime(cached_first, ns=(1, 1))

    with interface._open_chunk_file(str(second)) as cached_file:
        assert cached_file.read() == b"b" * 8

    assert not cached_first.exists()
    assert (cache_root / "second.zip").exists()


def test_open_chunk_file_bypasses_cache_for_non_zip(tmp_path):
    source = tmp_path / "feature.parquet"
    source.write_bytes(b"parquet")
    interface = PhysicalAIAVDatasetLocalInterface.__new__(
        PhysicalAIAVDatasetLocalInterface
    )
    interface.zip_cache_dir = None

    with interface._open_chunk_file(str(source)) as source_file:
        assert source_file.read() == b"parquet"
