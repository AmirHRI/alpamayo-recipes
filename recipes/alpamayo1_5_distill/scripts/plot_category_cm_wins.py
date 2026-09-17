"""Plot category-wise CM wins from saved NFE=1 trajectories without model inference."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from alpamayo.visualization.viz import project_waypoints_ftheta


CAMERA = "camera_front_wide_120fov"
COLOURS = {"CM 4B": "#d81b60", "CM 2B": "#1976d2", "Alpamayo 1.5": "#e69500"}


def load_run(path):
    with np.load(path, allow_pickle=False) as archive:
        clip_ids = archive["clip_ids"].astype(str)
        predictions = archive["pred_xyz"]
        truth = archive["gt_xyz"]
        if clip_ids.shape != (23331,) or len(np.unique(clip_ids)) != 23331:
            raise ValueError(f"{path}: expected 23,331 unique clips")
        if predictions.shape != (23331, 1, 6, 64, 3) or truth.shape != (23331, 64, 3):
            raise ValueError(f"{path}: unexpected trajectory shape")
        if not np.isfinite(predictions).all() or not np.isfinite(truth).all():
            raise ValueError(f"{path}: non-finite trajectories")
        if "inference_step" in archive and archive["inference_step"].item() != 1:
            raise ValueError(f"{path}: expected NFE=1")
        if "cameras" in archive and not np.array_equal(archive["cameras"], [1, 3]):
            raise ValueError(f"{path}: expected cameras [1, 3]")
    order = np.argsort(clip_ids)
    return clip_ids[order], predictions[order, 0], truth[order]


def draw_errors(predictions, truth):
    return np.linalg.norm(
        predictions[..., :2].astype(np.float64) - truth[..., :2].astype(np.float64), axis=-1
    ).mean(axis=-1)


def select_wins(scores, categories):
    candidates = scores.copy()
    candidates["category"] = categories.reindex(scores.index)
    if candidates["category"].isna().any() or candidates["category"].str.strip().eq("").any():
        raise ValueError("Missing scenario categories")
    candidates["improvement_m"] = candidates["Alpamayo 1.5"] - candidates["CM 4B"]
    winners = candidates.loc[candidates["improvement_m"] > 0].reset_index()
    winners = winners.sort_values(["category", "improvement_m", "clip_id"], ascending=[True, False, True])
    selected = winners.drop_duplicates("category").copy()
    missing = sorted(set(candidates["category"]) - set(selected["category"]))
    counts = candidates.groupby("category").size()
    win_counts = winners.groupby("category").size()
    selected["category_clips"] = selected["category"].map(counts)
    selected["winning_clips"] = selected["category"].map(win_counts)
    return selected, missing


def camera_calibration(intrinsics, extrinsics):
    camera_intrinsics = intrinsics.loc[CAMERA]
    camera_extrinsics = extrinsics.loc[CAMERA]
    calibration = [float(camera_intrinsics[key]) for key in (
        "width", "height", "cx", "cy", "fw_poly_0", "fw_poly_1", "fw_poly_2", "fw_poly_3", "fw_poly_4"
    )]
    rotation = Rotation.from_quat([camera_extrinsics[key] for key in ("qx", "qy", "qz", "qw")]).as_matrix()
    translation = np.array([camera_extrinsics[key] for key in ("x", "y", "z")], dtype=np.float64)
    return rotation, translation, calibration


def draw_example(record, image, truth, predictions, intrinsics, extrinsics, annotation):
    rotation, translation, calibration = camera_calibration(intrinsics, extrinsics)
    if image.shape[:2] != (int(calibration[1]), int(calibration[0])):
        raise ValueError("Camera image dimensions do not match calibration")
    figure, (camera_axis, bev_axis) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1.65, 1]})
    scores_text = " | ".join(f"{label}: {record[label]:.3f}" for label in predictions)
    figure.suptitle(
        f"{record['category']} | NFE=1 | min_ADE (m): {scores_text}\n"
        f"{record['clip_id']} | t0={annotation['t0_relative'] / 1e6:.1f}s | {annotation['nav_text']}\n"
        f"Selected largest front-visible CM 4B win: {record['improvement_m']:.3f} m; not a representative sample",
        fontsize=11,
    )
    camera_axis.imshow(image)
    projected_counts = {}
    for label, modes in predictions.items():
        colour = COLOURS[label]
        best = int(np.argmin(draw_errors(modes, truth[None])))
        for trajectory in modes:
            camera_points = project_waypoints_ftheta(trajectory.astype(np.float64), rotation, translation, calibration)
            if len(camera_points):
                camera_axis.plot(camera_points[:, 0], camera_points[:, 1], color=colour, alpha=0.15, lw=1)
            bev_axis.plot(-trajectory[:, 1], trajectory[:, 0], color=colour, alpha=0.2, lw=1)
        camera_points = project_waypoints_ftheta(modes[best].astype(np.float64), rotation, translation, calibration)
        projected_counts[label] = len(camera_points)
        if len(camera_points):
            camera_axis.plot(camera_points[:, 0], camera_points[:, 1], color=colour, lw=2.3, label=label)
        bev_axis.plot(-modes[best, :, 1], modes[best, :, 0], color=colour, lw=2.3, label=label)
    camera_points = project_waypoints_ftheta(truth.astype(np.float64), rotation, translation, calibration)
    projected_counts["GT"] = len(camera_points)
    if len(camera_points):
        camera_axis.plot(camera_points[:, 0], camera_points[:, 1], "--", color="#00b853", lw=2.5, label="GT")
    bev_axis.plot(-truth[:, 1], truth[:, 0], "--", color="#00b853", lw=2.5, label="GT")
    camera_axis.set_xlim(0, calibration[0])
    camera_axis.set_ylim(calibration[1], 0)
    camera_axis.axis("off")
    camera_axis.legend(loc="lower right", fontsize=9)
    camera_axis.set_title("Front-wide at t0 | bold: best of six (GT oracle); faint: all draws", fontsize=10)
    bev_axis.plot(0, 0, "k^", markersize=8)
    bev_axis.set_aspect("equal", adjustable="datalim")
    bev_axis.grid(alpha=0.25)
    bev_axis.set_xlabel("Lateral (m): left < 0, right > 0")
    bev_axis.set_ylabel("Forward (m)")
    bev_axis.set_title("BEV | equal metre scale | full saved trajectories", fontsize=10)
    bev_axis.legend(loc="best", fontsize=9)
    figure.text(0.02, 0.015, "Camera curves are clipped by the projection; the BEV includes all waypoints. No model inference.", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.88))
    return figure, projected_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=Path("/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests"))
    parser.add_argument("--local-dir", default="/temp/achahe/physical_ai_av")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    paths = {
        "CM 4B": args.training_root / "cm4b_availableval23331_fixedt0_stripped_ep2_nfe1_21166.npz",
        "CM 2B": args.training_root / "cm2b_availableval23331_fixedt0_stripped_ep2_nfe1_21167.npz",
        "Alpamayo 1.5": args.training_root / "teacher15_availableval23331_fixedt0_stripped_nfe1_21170.npz",
    }
    labels = pd.read_csv(args.manifest_dir / "lcdrive_val_primary_scenario_for_table2.csv", dtype=str)
    if labels["clip_uuid"].isna().any() or labels["clip_uuid"].duplicated().any():
        raise ValueError("Scenario labels require unique clip UUIDs")
    categories = labels.set_index("clip_uuid")["scenario_category"]
    annotations = json.loads((args.manifest_dir / "nav_lcdrive_val_available_fixedt0_23331_stripped.json").read_text())
    annotations = {row["clip_id"]: row for row in annotations}
    reference_ids, reference_gt = None, None
    predictions_by_model, scores_by_model = {}, {}
    for label, path in paths.items():
        clip_ids, predictions, truth = load_run(path)
        if reference_ids is None:
            reference_ids, reference_gt = clip_ids, truth
        else:
            np.testing.assert_array_equal(clip_ids, reference_ids)
            np.testing.assert_array_equal(truth, reference_gt)
        predictions_by_model[label] = predictions
        scores_by_model[label] = draw_errors(predictions, truth[:, None]).min(axis=1)
    if set(reference_ids) != set(annotations) or any(row["t0_relative"] != 5100000 for row in annotations.values()):
        raise ValueError("Annotations must match the full-validation clips and fixed t0")
    scores = pd.DataFrame(scores_by_model, index=pd.Index(reference_ids, name="clip_id"))
    selected, missing = select_wins(scores, categories)
    row_by_clip = {clip: position for position, clip in enumerate(reference_ids)}
    if not args.select_only:
        from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface

        interface = PhysicalAIAVDatasetLocalInterface(
            local_dir=args.local_dir, chunk_ids="0-3146", features_metadata="features.csv", clip_index_metadata="clip_index.parquet"
        )
        visible_records = []
        for record in selected.to_dict("records"):
            pool = scores.loc[categories.reindex(scores.index).eq(record["category"])].copy()
            pool["improvement_m"] = pool["Alpamayo 1.5"] - pool["CM 4B"]
            pool = pool.loc[pool["improvement_m"] > 0].reset_index().sort_values(
                ["improvement_m", "clip_id"], ascending=[False, True]
            )
            for candidate in pool.to_dict("records"):
                clip = candidate["clip_id"]
                position = row_by_clip[clip]
                truth = reference_gt[position]
                rotation, translation, calibration = camera_calibration(
                    interface.get_clip_feature(clip, "camera_intrinsics"),
                    interface.get_clip_feature(clip, "sensor_extrinsics"),
                )
                trajectories = [truth]
                for predictions in predictions_by_model.values():
                    modes = predictions[position]
                    trajectories.append(modes[int(np.argmin(draw_errors(modes, truth[None])))])
                if all(len(project_waypoints_ftheta(trajectory.astype(np.float64), rotation, translation, calibration)) >= 8 for trajectory in trajectories):
                    record.update(candidate)
                    record["nav_text"] = annotations[clip]["nav_text"]
                    visible_records.append(record)
                    break
            else:
                raise ValueError(f"No front-visible winning example for {record['category']}")
        selected = pd.DataFrame(visible_records)
    args.out.mkdir(parents=True, exist_ok=True)
    selected["t0_us"] = 5100000
    selected["nav_text"] = selected["clip_id"].map(lambda clip: annotations[clip]["nav_text"])
    selected.to_csv(args.out / "selected_examples.csv", index=False)
    (args.out / "provenance.json").write_text(json.dumps({
        "archives": {label: str(path) for label, path in paths.items()},
        "metric": "min_ADE: minimum over six time-averaged XY trajectory errors",
        "selection": "largest positive Alpamayo minus CM 4B min_ADE per category; ties by clip UUID; rendering requires at least 8 front-visible points for GT and each best-of-six trajectory",
        "nfe": 1, "cameras": [1, 3], "t0_us": 5100000,
        "categories_without_wins": missing,
    }, indent=2) + "\n")
    print(selected.to_string(index=False), flush=True)
    print(f"Selected {len(selected)} categories; categories without a CM 4B win: {missing}", flush=True)
    if args.select_only:
        return
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    front_camera = interface.features.CAMERA.CAMERA_FRONT_WIDE_120FOV
    projection_report = []
    with PdfPages(args.out / "category_examples.pdf") as pdf:
        for example_number, record in enumerate(selected.to_dict("records"), start=1):
            clip = record["clip_id"]
            sample = load_physical_aiavdataset(
                clip, t0_us=5100000, avdi=interface, num_history_steps=16,
                num_future_steps=64, time_step=0.1, camera_features=[front_camera], num_frames=1,
            )
            image = sample["image_frames"][0, 0].permute(1, 2, 0).numpy()
            if image.dtype != np.uint8:
                image = np.clip(image / 255.0 if image.max() > 1 else image, 0, 1)
            position = row_by_clip[clip]
            figure, projected = draw_example(
                record, image, reference_gt[position],
                {label: predictions[position] for label, predictions in predictions_by_model.items()},
                interface.get_clip_feature(clip, "camera_intrinsics"),
                interface.get_clip_feature(clip, "sensor_extrinsics"), annotations[clip],
            )
            filename = f"{example_number:02d}_{clip}.png"
            figure.savefig(args.out / filename, dpi=140)
            pdf.savefig(figure)
            plt.close(figure)
            projection_report.append({"category": record["category"], "clip_id": clip, "png": filename, "projected_best_points": projected})
            print(f"Rendered {record['category']}: {filename}; projected points {projected}", flush=True)
    (args.out / "plots.json").write_text(json.dumps(projection_report, indent=2) + "\n")
    print(f"Saved {len(projection_report)} camera/BEV figures and category_examples.pdf to {args.out}", flush=True)


if __name__ == "__main__":
    main()