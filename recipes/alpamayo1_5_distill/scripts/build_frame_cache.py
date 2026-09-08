#!/usr/bin/env python3
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

r"""Materialise the exact frames the nav 2-camera arm consumes, once, at full resolution.

WHY THIS EXISTS.  Measured on job 20710: the live pipeline streams **1.64 TiB of NFS per
epoch** to deliver 879,976 images.  Three amplification stages, all measured, none of them
avoidable by tuning:

    whole chunk ZIPs are copied   48.6 of ~98 clips per chunk are in the manifest   2.0x
    whole clip videos are read    604-frame 20.1 s 1080p mp4; ~27 frames are used    ~22x
    frames arrive full-res        1920x1080 decoded, then the ViT sees 576x320       11x

``/temp`` delivers ~120 MB/s aggregate (already ``nconnect=8``), so 1.64 TiB *is* ~4 h of wire
time per epoch.  That is exactly what the step timings showed: median 3.0 s/step against a
2.0 s GPU floor, and a tail long enough to trip the 120 s dataloader timeout three times.

The set of frames is a closed-form function of the manifest -- ``load_physical_aiavdataset``
picks ``[t0-0.3s, t0-0.2s, t0-0.1s, t0]`` from ``(clip_id, t0_relative)``, with no seed and no
epoch dependence -- so all 879,976 of them can be enumerated and written down ahead of time.
This script does that.  Afterwards a training job stages ~126 GiB into tmpfs once and does
**zero NFS reads** for the rest of the run.

FIDELITY.  Frames are taken from ``PAIDatasetWithNav`` itself, not from a reimplementation of
the frame maths -- an off-by-one frame here would silently poison every sample, and the KD
objective depends on teacher and student seeing element-wise identical ``input_ids``.  The one
liberty taken is memoising ``get_clip_feature`` across the anchors of a clip (see
``_ClipFeatureMemo``); ``--verify`` proves that memoised samples are bit-identical to
freshly-loaded ones before any bulk work starts.

STORAGE.  Each anchor's 4 frames per camera go into one tiny mp4 rather than 4 JPEGs.  They are
0.1 s apart and highly correlated, so inter-frame prediction halves the size at higher quality:
measured over 6 anchors, x264 crf18 is 1,202 KB/anchor (126 GB total) against 2,538 KB for JPEG
q90 4:2:0 (266 GB).  For scale, the source video is ~8 KB/frame, so crf18 at ~150 KB/frame sits
far above the quality the pipeline currently sees.

Layout, one ZIP per clip so staging moves 32k files rather than 880k::

    <out>/<chunk:04d>/<clip_id>.zip
        <t0_us>.cam<idx>.mp4    4 frames, 1920x1080, one per camera
        <t0_us>.ego.npz         ego/timestamp tensors for that anchor
        _manifest.json          {version, cameras, crf, anchors: [t0_us, ...]}

``nav_text`` is deliberately NOT cached: it comes from the annotation JSON at train time, so
``strip_nav_turn_distance`` and future nav-text edits do not invalidate 126 GB of video.

Usage::

    build_frame_cache.py --out /temp/achahe/physical_ai_av/framecache_nav2cam_1080p --verify 8
    build_frame_cache.py --out ... --shard 0 --num-shards 32
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile

import numpy as np

# The recipe tree is not installed as a package for a bare script run.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "recipes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from alpamayo.data.pai_nav import PAIDatasetWithNav  # noqa: E402

#: Loader camera order, mirrored from ``camera_subset._DEFAULT_CAMERA_FEATURES``. The arm uses
#: [1, 3] = front-wide + front-telephoto; the indices are stored in the cache so a reader can
#: assert it was built for the camera set it is about to train on.
_DEFAULT_CAMERA_FEATURES = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

CACHE_VERSION = 1

#: Tensors that are a pure function of (clip_id, t0_us) and therefore cacheable. ``nav_text``,
#: ``clip_id`` and ``t0_us`` come from the annotation JSON at read time instead.
_EGO_KEYS = (
    "ego_history_xyz",
    "ego_history_rot",
    "ego_future_xyz",
    "ego_future_rot",
    "camera_indices",
    "relative_timestamps",
    "absolute_timestamps",
)


class _ClipFeatureMemo:
    """Hold one clip's decoders so its anchors do not re-read the member from NFS.

    ``get_clip_feature`` opens the chunk ZIP and reads the whole member on every call, so
    iterating anchors naively would pull each ~10 MB clip video 3.44 times (the mean anchors
    per clip) -- 2.2 TB for the pass, worse than the pipeline being replaced. Anchors are
    visited grouped by clip, so a one-clip memo turns that back into one read per (clip,
    camera).

    The memo is keyed by clip and dropped whole when the clip changes, which bounds it at three
    entries (two cameras plus egomotion) and keeps at most one clip's video resident.
    ``SeekVideoReader`` is seek-addressed and builds its keyframe index lazily, so reuse across
    anchors is both correct and strictly cheaper -- ``--verify`` checks the "correct" half
    against freshly-loaded samples rather than asserting it.
    """

    def __init__(self, avdi) -> None:
        self._avdi = avdi
        self._inner = avdi.get_clip_feature
        self._clip_id: str | None = None
        self._entries: dict[str, object] = {}
        self.hits = 0
        self.misses = 0

    def install(self) -> None:
        self._avdi.get_clip_feature = self  # type: ignore[method-assign]

    def uninstall(self) -> None:
        self._avdi.get_clip_feature = self._inner  # type: ignore[method-assign]
        self.reset()

    def reset(self) -> None:
        for entry in self._entries.values():
            close = getattr(entry, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # a reader that will not close must not fail the pass
                    pass
        self._entries.clear()
        self._clip_id = None

    def __call__(self, clip_id: str, feature: str, maybe_stream: bool = False):
        if clip_id != self._clip_id:
            self.reset()
            self._clip_id = clip_id
        if feature in self._entries:
            self.hits += 1
            return self._entries[feature]
        self.misses += 1
        value = self._inner(clip_id, feature, maybe_stream=maybe_stream)
        self._entries[feature] = value
        return value


def build_dataset(args) -> PAIDatasetWithNav:
    """Construct the base dataset exactly as ``CameraSubsetPAIDataset`` does.

    ``model_config=None`` / ``vla_preprocess_args=None`` is what the wrapper passes so the raw
    sample comes back unpreprocessed, and assigning ``camera_features`` after construction is
    the same one-line camera subsetting the wrapper performs -- the base then reads only the
    requested cameras rather than decoding four and discarding two.
    """
    ds = PAIDatasetWithNav(
        annotations_path=args.manifest,
        local_dir=args.local_dir,
        chunk_ids=args.chunk_ids,
        clip_uuid_filter=args.clip_filter,
        model_config=None,
        vla_preprocess_args=None,
    )
    ds.camera_features = [_DEFAULT_CAMERA_FEATURES[c] for c in args.cameras]
    return ds


def encode_clip_mp4(frames: torch.Tensor, crf: int) -> bytes:
    """Encode ``(T, 3, H, W)`` uint8 as a single-GOP mp4 and return the container bytes.

    One keyframe followed by P-frames: the four frames are 0.1 s apart, which is what makes
    this half the size of the equivalent JPEGs. ``yuv420p`` matches the source's own pixel
    format, so the colour conversion is the same one the decode path already performs.
    """
    import av

    n_frames, _, height, width = frames.shape
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        # gop_size beyond the frame count keeps the group at one I-frame; without it x264 is
        # free to insert more and the inter-frame saving disappears.
        stream.gop_size = n_frames * 8
        stream.options = {"crf": str(crf)}
        array = frames.permute(0, 2, 3, 1).contiguous().numpy()
        for i in range(n_frames):
            frame = av.VideoFrame.from_ndarray(array[i], format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def decode_clip_mp4(payload: bytes) -> torch.Tensor:
    """Inverse of :func:`encode_clip_mp4`; returns ``(T, 3, H, W)`` uint8."""
    import av

    with av.open(io.BytesIO(payload)) as container:
        frames = [
            torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
            for frame in container.decode(video=0)
        ]
    return torch.stack(frames)


def anchor_entry_names(t0_us: int, cameras: list[int]) -> list[str]:
    return [f"{t0_us}.cam{c}.mp4" for c in cameras] + [f"{t0_us}.ego.npz"]


def zip_is_complete(path: str, t0s: list[int], cameras: list[int]) -> bool:
    """A ZIP counts as done only if its manifest and every expected member are present."""
    if not os.path.exists(path):
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            if "_manifest.json" not in names:
                return False
            manifest = json.loads(zf.read("_manifest.json"))
            if manifest.get("version") != CACHE_VERSION:
                return False
            if list(manifest.get("cameras", [])) != list(cameras):
                return False
            if sorted(manifest.get("anchors", [])) != sorted(t0s):
                return False
            for t0 in t0s:
                if not set(anchor_entry_names(t0, cameras)) <= names:
                    return False
    except (zipfile.BadZipFile, KeyError, ValueError, OSError):
        return False
    return True


def write_clip_zip(path: str, payloads: dict[str, bytes], manifest: dict) -> int:
    """Write one clip's members atomically; a killed shard never leaves a half ZIP behind."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = f"{path}.partial.{os.getpid()}"
    try:
        # ZIP_STORED throughout: mp4 and npz payloads are already compressed, so deflate would
        # burn CPU on the write and again on every training read for no size win.
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
            for name, payload in payloads.items():
                zf.writestr(name, payload)
        size = os.path.getsize(partial)
        os.replace(partial, path)
        return size
    finally:
        try:
            os.unlink(partial)
        except FileNotFoundError:
            pass


def pack_anchor(sample: dict, cameras: list[int], crf: int) -> dict[str, bytes]:
    """Turn one raw sample into the members that represent it in the cache."""
    t0_us = int(sample["t0_us"])
    images = sample["image_frames"]  # (n_cam, n_frames, 3, H, W) uint8
    if images.shape[0] != len(cameras):
        raise RuntimeError(
            f"expected {len(cameras)} cameras in the sample, got {images.shape[0]}; "
            "camera_features was not applied"
        )
    payloads = {
        f"{t0_us}.cam{cam}.mp4": encode_clip_mp4(images[i], crf)
        for i, cam in enumerate(cameras)
    }
    buffer = io.BytesIO()
    np.savez(
        buffer,
        **{k: sample[k].numpy() for k in _EGO_KEYS if k in sample},
    )
    payloads[f"{t0_us}.ego.npz"] = buffer.getvalue()
    return payloads


def verify(ds, args, n: int) -> None:
    """Prove the memo and the codec before spending hours on the bulk pass.

    Two independent claims are checked per sampled anchor: that memoised loading returns the
    tensors a cold dataset returns (bit-identical, since a wrong frame index would be invisible
    downstream), and that an encode/decode round trip stays close enough to the source that the
    ViT cannot tell -- reported as PSNR rather than asserted, so the number lands in the job log
    next to the checkpoint it produced.
    """
    rng = np.random.default_rng(0)
    indices = sorted(rng.choice(len(ds), size=min(n, len(ds)), replace=False).tolist())
    print(f"[verify] {len(indices)} anchors, memoised vs cold load + codec round trip", flush=True)

    memo = _ClipFeatureMemo(ds.avdi)
    psnrs: list[float] = []
    for idx in indices:
        cold = ds[idx]
        memo.install()
        try:
            # Load the clip's other anchors first so the memo is genuinely warm for this one,
            # which is the state the bulk pass will read in.
            clip_id = ds._samples[idx]["clip_id"]
            for j, entry in enumerate(ds._samples):
                if entry["clip_id"] == clip_id and j != idx:
                    ds[j]
                    break
            warm = ds[idx]
        finally:
            memo.uninstall()

        for key in ("image_frames",) + _EGO_KEYS:
            if key not in cold:
                continue
            if not torch.equal(cold[key], warm[key]):
                raise SystemExit(
                    f"[verify] FAIL: anchor {idx} key {key!r} differs between memoised and "
                    "cold load; the memo is not transparent, do not build the cache"
                )

        for i in range(cold["image_frames"].shape[0]):
            source = cold["image_frames"][i]
            back = decode_clip_mp4(encode_clip_mp4(source, args.crf))
            if back.shape != source.shape:
                raise SystemExit(
                    f"[verify] FAIL: round trip changed shape {tuple(source.shape)} -> "
                    f"{tuple(back.shape)}"
                )
            mse = torch.mean((source.float() - back.float()) ** 2).item()
            psnrs.append(float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse))

    finite = [p for p in psnrs if np.isfinite(p)]
    print(
        f"[verify] OK: memoised == cold on {len(indices)} anchors "
        f"({len(_EGO_KEYS) + 1} tensors each)",
        flush=True,
    )
    if finite:
        print(
            f"[verify] crf{args.crf} round trip PSNR: min {min(finite):.1f} dB  "
            f"mean {sum(finite) / len(finite):.1f} dB  (>40 dB is visually lossless)",
            flush=True,
        )


def finalize(ds, args) -> None:
    """Scan the built cache once and write the index the reader opens at startup.

    Without this the reader would have to stat 32k ZIPs to learn which anchors exist, on every
    rank, at every job start. The index also pins the build's identity -- camera set and crf --
    so a reader can refuse a cache built for a different camera subset rather than silently
    training on the wrong pixels.
    """
    clip_ids = [entry["clip_id"] for entry in ds._samples]
    chunks = ds.avdi.clip_index.loc[clip_ids, "chunk"].to_numpy()
    by_clip: dict[tuple[int, str], list[int]] = {}
    for idx, (chunk, clip_id) in enumerate(zip(chunks, clip_ids)):
        by_clip.setdefault((int(chunk), str(clip_id)), []).append(idx)

    clips: dict[str, int] = {}
    complete = incomplete = 0
    anchors = 0
    for (chunk, clip_id), indices in sorted(by_clip.items()):
        t0s = [int(ds._samples[i]["t0_relative"]) for i in indices]
        path = os.path.join(args.out, f"{chunk:04d}", f"{clip_id}.zip")
        if zip_is_complete(path, t0s, args.cameras):
            clips[clip_id] = chunk
            complete += 1
            anchors += len(indices)
        else:
            incomplete += 1

    index = {
        "version": CACHE_VERSION,
        "cameras": args.cameras,
        "camera_features": [_DEFAULT_CAMERA_FEATURES[c] for c in args.cameras],
        "crf": args.crf,
        "manifest": os.path.abspath(args.manifest),
        "clips": clips,
        "anchors": anchors,
    }
    path = os.path.join(args.out, "_index.json")
    partial = f"{path}.partial.{os.getpid()}"
    with open(partial, "w") as handle:
        json.dump(index, handle)
    os.replace(partial, path)
    print(
        f"[cache] index written: {complete} clips complete, {incomplete} missing, "
        f"{anchors}/{len(ds)} anchors -> {path}",
        flush=True,
    )
    if incomplete:
        raise SystemExit(
            f"[cache] {incomplete} clips are missing or incomplete; re-run the build "
            "(it skips finished clips) before training on this cache"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--manifest",
        default="/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/"
        "nav_lcdrive_train_anchors_all.json",
    )
    parser.add_argument("--local-dir", default="/temp/achahe/physical_ai_av/")
    parser.add_argument(
        "--clip-filter",
        default="/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/"
        "lcdrive_train_clip_uuids.txt",
    )
    parser.add_argument("--chunk-ids", default="0-3146")
    parser.add_argument("--cameras", default="1,3", help="loader camera indices")
    parser.add_argument("--out", required=True)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--verify", type=int, default=0, metavar="N")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="run --verify and exit without writing, whatever the shard settings say",
    )
    parser.add_argument("--limit-clips", type=int, default=0, help="0 = no limit; for smoke runs")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="scan the built cache and write _index.json; exits non-zero if anything is missing",
    )
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    args.cameras = [int(c) for c in args.cameras.split(",")]
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard {args.shard} out of range for --num-shards {args.num_shards}")

    started = time.time()
    ds = build_dataset(args)
    print(
        f"[cache] {len(ds)} anchors, cameras {args.cameras} "
        f"({[_DEFAULT_CAMERA_FEATURES[c] for c in args.cameras]}), crf={args.crf}",
        flush=True,
    )

    if args.verify:
        verify(ds, args, args.verify)
        if args.verify_only:
            print("[cache] verify-only run complete; re-run sharded to build", flush=True)
            return

    if args.finalize:
        finalize(ds, args)
        return

    # Group anchors by clip, and clips by chunk: one chunk's ZIP members stay hot in the page
    # cache while its ~49 in-manifest clips are drained, which is the same locality the live
    # loader's grouped sampler buys, applied to a single sequential pass.
    clip_ids = [entry["clip_id"] for entry in ds._samples]
    chunks = ds.avdi.clip_index.loc[clip_ids, "chunk"].to_numpy()
    by_clip: dict[tuple[int, str], list[int]] = {}
    for idx, (chunk, clip_id) in enumerate(zip(chunks, clip_ids)):
        by_clip.setdefault((int(chunk), str(clip_id)), []).append(idx)

    # Shard on the chunk so a shard owns whole chunks and two shards never contend for the
    # same multi-GB ZIP.
    ordered = sorted(by_clip)
    mine = [key for key in ordered if key[0] % args.num_shards == args.shard]
    if args.limit_clips:
        mine = mine[: args.limit_clips]
    total_anchors = sum(len(by_clip[k]) for k in mine)
    print(
        f"[cache] shard {args.shard}/{args.num_shards}: {len(mine)} clips, "
        f"{total_anchors} anchors, out={args.out}",
        flush=True,
    )

    memo = _ClipFeatureMemo(ds.avdi)
    memo.install()
    done_clips = skipped = written_anchors = 0
    written_bytes = 0
    try:
        for n, key in enumerate(mine, 1):
            chunk, clip_id = key
            indices = by_clip[key]
            t0s = [int(ds._samples[i]["t0_relative"]) for i in indices]
            path = os.path.join(args.out, f"{chunk:04d}", f"{clip_id}.zip")
            if zip_is_complete(path, t0s, args.cameras):
                skipped += 1
                continue

            payloads: dict[str, bytes] = {}
            for idx in indices:
                sample = ds[idx]
                if sample is None:
                    raise RuntimeError(f"dataset returned None for anchor {idx} ({clip_id})")
                payloads.update(pack_anchor(sample, args.cameras, args.crf))
            memo.reset()

            written_bytes += write_clip_zip(
                path,
                payloads,
                {
                    "version": CACHE_VERSION,
                    "cameras": args.cameras,
                    "camera_features": [_DEFAULT_CAMERA_FEATURES[c] for c in args.cameras],
                    "crf": args.crf,
                    "clip_id": clip_id,
                    "chunk": chunk,
                    "anchors": sorted(t0s),
                },
            )
            done_clips += 1
            written_anchors += len(indices)

            if n % args.log_every == 0 or n == len(mine):
                elapsed = time.time() - started
                rate = done_clips / elapsed if elapsed else 0.0
                remaining = (len(mine) - n) / rate / 3600 if rate else float("nan")
                print(
                    f"[cache] {n}/{len(mine)} clips "
                    f"(written {done_clips}, skipped {skipped}, {written_anchors} anchors, "
                    f"{written_bytes / 1024**3:.1f} GiB) "
                    f"{rate * 3600:.0f} clips/h, ~{remaining:.1f} h left",
                    flush=True,
                )
    finally:
        memo.uninstall()

    elapsed = time.time() - started
    print(
        f"[cache] shard {args.shard} done in {elapsed / 3600:.2f} h: "
        f"{done_clips} clips written, {skipped} already present, "
        f"{written_anchors} anchors, {written_bytes / 1024**3:.1f} GiB, "
        f"memo hits {memo.hits} / misses {memo.misses}",
        flush=True,
    )


if __name__ == "__main__":
    main()
