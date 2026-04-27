#!/usr/bin/env python3
"""Run raw Mega-SAM on the local Xperience-10M sample and compare with HDF5 GT."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parent
MEGASAM_ROOT = REPO_ROOT / "mega-sam"
sys.path.append(str(REPO_ROOT))

from lerobot_v3_common import (  # noqa: E402
    log_pose,
    log_static_trajectory,
    matrix_to_tum_row,
    match_by_timestamp,
    rr_image,
    rows_by_time_key,
    save_tum,
    set_rerun_time,
    trajectory_stats,
    tum_row_to_matrix,
    world_mats_to_relative_tum,
    write_intrinsics_txt,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "xperience-10m-sample"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tasks" / "xperience_10m_sample_megasam"
DEFAULT_PYTHON = Path.home() / "miniconda3" / "envs" / "mega_sam" / "bin" / "python"
DEFAULT_WEIGHTS = MEGASAM_ROOT / "checkpoints" / "megasam_final.pth"

GT_COLOR = [0, 200, 255]
MEGASAM_COLOR = [120, 220, 120]
MEGASAM_ALIGNED_COLOR = [255, 180, 40]


def sanitize_scene_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "xperience_sequence"


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def clear_files(directory: Path, patterns: tuple[str, ...]) -> None:
    if not directory.exists():
        return
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def read_video_info(path: Path) -> dict[str, int | float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def xperience_fps(root: h5py.File, fallback: float) -> float:
    if "video/frame_number" in root and "video/length_sec" in root:
        frame_count = int(root["video/frame_number"].shape[0])
        length_sec = float(root["video/length_sec"][()])
        if frame_count > 0 and length_sec > 0:
            return frame_count / length_sec
    return float(fallback or 20.0)


def select_frames(
    total_frames: int,
    *,
    start_frame: int,
    end_frame: int | None,
    stride: int,
    max_frames: int | None,
) -> list[int]:
    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    end_value = total_frames if end_frame is None else min(int(end_frame), total_frames)
    if end_value <= start_frame:
        raise ValueError(f"Empty frame range: start={start_frame}, end={end_value}")
    frames = list(range(int(start_frame), end_value, int(stride)))
    if max_frames is not None and max_frames > 0:
        frames = frames[: int(max_frames)]
    if not frames:
        raise ValueError("No frames selected")
    return frames


def calibration_matrix(root: h5py.File, key: str) -> np.ndarray:
    values = np.asarray(root[key][()], dtype=np.float64)
    if values.shape == (3, 3):
        return values
    if values.size == 4:
        fx, fy, cx, cy = values.reshape(-1)
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 0] = fx
        matrix[1, 1] = fy
        matrix[0, 2] = cx
        matrix[1, 2] = cy
        return matrix
    raise ValueError(f"Unsupported calibration shape for {key}: {values.shape}")


def pose_matrix_from_wxyz_translation(translation: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    quat_xyzw /= np.linalg.norm(quat_xyzw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    transform[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    return transform


def load_gt_world_from_camera(
    root: h5py.File,
    frame_ids: list[int],
    *,
    pose_convention: str,
    camera_extrinsics_key: str,
    camera_extrinsics_convention: str,
) -> np.ndarray:
    translations = np.asarray(root["slam/trans_xyz"][frame_ids], dtype=np.float64)
    quats_wxyz = np.asarray(root["slam/quat_wxyz"][frame_ids], dtype=np.float64)
    poses = np.stack(
        [
            pose_matrix_from_wxyz_translation(translation, quat_wxyz)
            for translation, quat_wxyz in zip(translations, quats_wxyz)
        ],
        axis=0,
    )
    if pose_convention == "camera_from_world":
        poses = np.linalg.inv(poses)
    elif pose_convention != "world_from_camera":
        raise ValueError(f"Unsupported --gt-pose-convention: {pose_convention}")

    if camera_extrinsics_key and camera_extrinsics_key.lower() != "none":
        if camera_extrinsics_key not in root:
            raise KeyError(f"Missing camera extrinsics key in HDF5: {camera_extrinsics_key}")
        extrinsic = np.asarray(root[camera_extrinsics_key][()], dtype=np.float64)
        if extrinsic.shape != (4, 4):
            raise ValueError(f"Camera extrinsics must be 4x4, got {extrinsic.shape}: {camera_extrinsics_key}")
        if camera_extrinsics_convention == "camera_from_body":
            poses = poses @ np.linalg.inv(extrinsic)
        elif camera_extrinsics_convention == "body_from_camera":
            poses = poses @ extrinsic
        else:
            raise ValueError(f"Unsupported --gt-camera-extrinsics-convention: {camera_extrinsics_convention}")
    return poses


def extract_frames(video_path: Path, frame_ids: list[int], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_files(output_dir, ("*.png", "*.jpg", "*.jpeg"))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    written: list[Path] = []
    desired = [int(frame_id) for frame_id in frame_ids]
    next_id = 0
    current = desired[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, current)
    try:
        while next_id < len(desired):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if current == desired[next_id]:
                output_path = output_dir / f"{next_id:06d}.png"
                if not cv2.imwrite(str(output_path), frame_bgr):
                    raise IOError(f"Failed to write frame: {output_path}")
                written.append(output_path)
                next_id += 1
                while next_id < len(desired) and desired[next_id] < current + 1:
                    next_id += 1
            current += 1
    finally:
        cap.release()

    if len(written) != len(frame_ids):
        raise RuntimeError(f"Extracted {len(written)}/{len(frame_ids)} requested frames from {video_path}")
    return written


def prepare_metric_depth(
    root: h5py.File,
    frame_ids: list[int],
    output_dir: Path,
    *,
    fov_degrees: float,
    depth_scale: float,
    depth_min: float,
    depth_max: float,
    confidence_min: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_files(output_dir, ("*.npz",))

    depth_data = root["depth/depth"]
    confidence_data = root["depth/confidence"] if "depth/confidence" in root else None
    for output_index, source_frame in enumerate(frame_ids):
        depth = np.asarray(depth_data[int(source_frame)], dtype=np.float32) * float(depth_scale)
        valid = np.isfinite(depth) & (depth >= float(depth_min)) & (depth <= float(depth_max))
        depth = np.where(valid, depth, 0.0).astype(np.float32)
        payload: dict[str, Any] = {
            "depth": depth,
            "fov": np.asarray(float(fov_degrees), dtype=np.float32),
            "source_frame": np.asarray(int(source_frame), dtype=np.int32),
        }
        if confidence_data is not None and confidence_min > 0:
            confidence = np.asarray(confidence_data[int(source_frame)], dtype=np.uint8)
            payload["stereo_valid"] = (valid & (confidence >= int(confidence_min))).astype(np.uint8)
            payload["confidence"] = confidence
        np.savez(output_dir / f"{output_index:06d}.npz", **payload)
    return len(frame_ids)


def remove_megasam_scene_outputs(scene_name: str) -> None:
    npz_path = MEGASAM_ROOT / "outputs" / f"{scene_name}_droid.npz"
    if npz_path.exists():
        npz_path.unlink()
    reconstruction_dir = MEGASAM_ROOT / "reconstructions" / scene_name
    if reconstruction_dir.exists():
        shutil.rmtree(reconstruction_dir)


def run_megasam_tracking(
    args: argparse.Namespace,
    *,
    image_dir: Path,
    metric_root: Path,
    intrinsics_path: Path,
    scene_name: str,
    frame_count: int,
) -> Path:
    command = [
        *shlex.split(args.python),
        "camera_tracking_scripts/test_demo.py",
        "--datapath",
        str(image_dir),
        "--weights",
        str(args.weights.expanduser().resolve()),
        "--scene_name",
        scene_name,
        "--mono_depth_path",
        str(args.output / "Depth-Anything" / "video_visualization"),
        "--metric_depth_path",
        str(metric_root),
        "--intrinsics",
        str(intrinsics_path),
        "--depth_source",
        "metric",
        "--disable_vis",
        "--buffer",
        str(max(int(args.buffer), int(frame_count))),
        "--beta",
        str(args.beta),
        "--filter_thresh",
        str(args.filter_thresh),
        "--warmup",
        str(args.warmup),
        "--keyframe_thresh",
        str(args.keyframe_thresh),
        "--frontend_thresh",
        str(args.frontend_thresh),
        "--frontend_window",
        str(args.frontend_window),
        "--frontend_radius",
        str(args.frontend_radius),
        "--frontend_nms",
        str(args.frontend_nms),
        "--backend_thresh",
        str(args.backend_thresh),
        "--backend_radius",
        str(args.backend_radius),
        "--backend_nms",
        str(args.backend_nms),
    ]
    if args.upsample:
        command.append("--upsample")
    if args.megasam_mask_vis:
        command.extend(["--mask_vis_path", str(args.output / "megasam_valid_mask_vis")])
    if args.megasam_motion_mask_vis:
        command.extend(["--motion_mask_vis_path", str(args.output / "megasam_motion_mask_vis")])
    if not (args.enable_megasam_full_ba or args.megasam_motion_mask_vis):
        command.append("--disable_full_ba")
    run_command(command, cwd=MEGASAM_ROOT, dry_run=args.dry_run)
    return MEGASAM_ROOT / "outputs" / f"{scene_name}_droid.npz"


def load_megasam_cam_c2w(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"Mega-SAM output not found: {npz_path}")
    with np.load(npz_path) as data:
        if "cam_c2w" not in data:
            raise KeyError(f"{npz_path} does not contain cam_c2w")
        cam_c2w = np.asarray(data["cam_c2w"], dtype=np.float64)
    if cam_c2w.ndim != 3 or cam_c2w.shape[1:] != (4, 4):
        raise ValueError(f"cam_c2w must have shape Nx4x4, got {cam_c2w.shape}")
    return cam_c2w


def estimate_sim3_to_reference(
    reference_rows: np.ndarray,
    estimate_rows: np.ndarray,
    max_time_diff: float,
    *,
    min_displacement: float,
) -> dict[str, Any]:
    ref, est = match_by_timestamp(reference_rows, estimate_rows, max_time_diff)
    if len(ref) < 3:
        raise ValueError("not enough timestamp matches to estimate Sim3")

    finite = np.isfinite(ref[:, 1:4]).all(axis=1) & np.isfinite(est[:, 1:4]).all(axis=1)
    ref = ref[finite]
    est = est[finite]
    if len(ref) < 3:
        raise ValueError("not enough finite matched positions to estimate Sim3")

    reference_origin = ref[0, 1:4].copy()
    estimate_origin = est[0, 1:4].copy()
    min_displacement = max(float(min_displacement), 0.0)
    ref_displacements = np.linalg.norm(ref[:, 1:4] - reference_origin, axis=1)
    est_displacements = np.linalg.norm(est[:, 1:4] - estimate_origin, axis=1)
    valid = (ref_displacements > min_displacement) & (est_displacements > min_displacement)
    if int(np.count_nonzero(valid)) >= 3:
        ref_used = ref[valid]
        est_used = est[valid]
    elif min_displacement <= 0.0:
        ref_used = ref
        est_used = est
    else:
        raise ValueError("not enough non-zero matched displacements to estimate Sim3")

    estimate_points = est_used[:, 1:4]
    reference_points = ref_used[:, 1:4]
    estimate_mean = estimate_points.mean(axis=0)
    reference_mean = reference_points.mean(axis=0)
    estimate_centered = estimate_points - estimate_mean
    reference_centered = reference_points - reference_mean
    estimate_variance = float(np.mean(np.sum(estimate_centered * estimate_centered, axis=1)))
    if estimate_variance <= 1e-12:
        raise ValueError("estimated trajectory variance is too small to align")

    covariance = (reference_centered.T @ estimate_centered) / len(estimate_points)
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    scale = float(np.sum(singular_values * np.diag(correction)) / estimate_variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid Sim3 scale estimate: {scale}")
    translation = reference_mean - scale * (rotation @ estimate_mean)

    residuals = reference_points - (scale * (estimate_points @ rotation.T) + translation)
    residual_norms = np.linalg.norm(residuals, axis=1)
    return {
        "enabled": True,
        "type": "sim3",
        "method": "umeyama_positions",
        "scale": scale,
        "rotation_matrix": rotation.tolist(),
        "rotation_angle_deg": float(np.degrees(R.from_matrix(rotation).magnitude())),
        "translation_xyz": translation.tolist(),
        "matched_frames": int(len(ref)),
        "used_samples": int(len(estimate_points)),
        "residual_rmse_m": float(np.sqrt(np.mean(residual_norms**2))),
        "residual_mean_m": float(np.mean(residual_norms)),
        "residual_max_m": float(np.max(residual_norms)),
        "origin_timestamp": float(est[0, 0]),
        "reference_origin_xyz": reference_origin.tolist(),
        "estimate_origin_xyz": estimate_origin.tolist(),
    }


def apply_sim3_to_rows(
    rows: np.ndarray,
    *,
    scale: float,
    rotation_matrix: list[list[float]] | np.ndarray,
    translation_xyz: list[float] | np.ndarray,
) -> np.ndarray:
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    translation = np.asarray(translation_xyz, dtype=np.float64)
    aligned_rows = []
    for row in np.asarray(rows, dtype=np.float64):
        pose = tum_row_to_matrix(row)
        pose[:3, 3] = translation + float(scale) * (rotation @ pose[:3, 3])
        pose[:3, :3] = rotation @ pose[:3, :3]
        aligned_rows.append(matrix_to_tum_row(float(row[0]), pose))
    return np.asarray(aligned_rows, dtype=np.float64)


def init_rerun(args: argparse.Namespace, save_rrd: Path | None) -> Any:
    import rerun as rr

    rr.init(args.rerun_app_id, spawn=args.rerun_spawn and save_rrd is None)
    if save_rrd is not None:
        save_rrd.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(save_rrd))
    try:
        view_coordinates = getattr(
            rr.ViewCoordinates,
            "RIGHT_HAND_Y_DOWN",
            rr.ViewCoordinates.RIGHT_HAND_Y_UP,
        )
        rr.log("world", view_coordinates, static=True)
    except Exception as exc:  # pragma: no cover
        print(f"Rerun coordinate setup failed, using defaults: {exc}")
    return rr


def log_rerun_image_file(rr: Any, entity: str, path: Path) -> bool:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return False
    rr.log(entity, rr_image(rr, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
    return True


def visualize_in_rerun(
    args: argparse.Namespace,
    *,
    image_dir: Path,
    frame_ids: list[int],
    timestamps: np.ndarray,
    gt_rows: np.ndarray,
    megasam_rows: np.ndarray,
    aligned_rows: np.ndarray | None,
    save_rrd: Path | None,
) -> None:
    if args.no_rerun:
        return

    rr = init_rerun(args, save_rrd)
    log_static_trajectory(rr, "world/xperience_slam_gt", gt_rows, GT_COLOR, args.rerun_trajectory_radius)
    log_static_trajectory(rr, "world/megasam", megasam_rows, MEGASAM_COLOR, args.rerun_trajectory_radius)
    if aligned_rows is not None:
        log_static_trajectory(
            rr,
            "world/megasam_sim3_aligned",
            aligned_rows,
            MEGASAM_ALIGNED_COLOR,
            args.rerun_trajectory_radius,
        )

    gt_lookup = rows_by_time_key(gt_rows)
    megasam_lookup = rows_by_time_key(megasam_rows)
    aligned_lookup = rows_by_time_key(aligned_rows) if aligned_rows is not None else {}
    max_frames = len(frame_ids) if args.max_rerun_frames is None else min(len(frame_ids), args.max_rerun_frames)
    for index in range(max_frames):
        timestamp = float(timestamps[index])
        set_rerun_time(rr, int(frame_ids[index]), timestamp)
        image_bgr = cv2.imread(str(image_dir / f"{index:06d}.png"), cv2.IMREAD_COLOR)
        if image_bgr is not None:
            rr.log("world/input/stereo_left", rr_image(rr, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
        if args.megasam_mask_vis:
            mask_root = args.output / "megasam_valid_mask_vis"
            log_rerun_image_file(rr, "world/input/megasam_valid_mask", mask_root / "raw" / f"{index:06d}.png")
            log_rerun_image_file(
                rr,
                "world/input/megasam_valid_mask_overlay",
                mask_root / "overlay" / f"{index:06d}.png",
            )
        if args.megasam_motion_mask_vis:
            mask_root = args.output / "megasam_motion_mask_vis"
            log_rerun_image_file(rr, "world/input/megasam_motion_mask", mask_root / "raw" / f"{index:06d}.png")
            log_rerun_image_file(
                rr,
                "world/input/megasam_motion_mask_overlay",
                mask_root / "overlay" / f"{index:06d}.png",
            )
        row_key = int(round(timestamp * 1_000_000.0))
        log_pose(rr, "world/xperience_slam_gt/current", gt_lookup.get(row_key), GT_COLOR, args.rerun_axis_length, args.rerun_origin_radius)
        log_pose(
            rr,
            "world/megasam/current",
            megasam_lookup.get(row_key),
            MEGASAM_COLOR,
            args.rerun_axis_length,
            args.rerun_origin_radius,
        )
        if aligned_rows is not None:
            log_pose(
                rr,
                "world/megasam_sim3_aligned/current",
                aligned_lookup.get(row_key),
                MEGASAM_ALIGNED_COLOR,
                args.rerun_axis_length,
                args.rerun_origin_radius,
            )
    print(f"Sent Rerun visualization: {max_frames} frames")
    if save_rrd is not None:
        print(f"Saved Rerun recording: {save_rrd}")


def write_report(
    output_dir: Path,
    *,
    scene_name: str,
    data_root: Path,
    frame_ids: list[int],
    fps: float,
    intrinsics: np.ndarray,
    fov_degrees: float,
    paths: dict[str, str],
    raw_stats: dict[str, float | int] | None,
    aligned_stats: dict[str, float | int] | None,
    sim3_alignment: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    report = {
        "scene_name": scene_name,
        "data_root": str(data_root),
        "fps": float(fps),
        "frame_count": int(len(frame_ids)),
        "source_frame_start": int(frame_ids[0]),
        "source_frame_end": int(frame_ids[-1]),
        "source_frame_stride": int(args.stride),
        "intrinsics": intrinsics.tolist(),
        "fov_degrees": float(fov_degrees),
        "gt_source": "annotation.hdf5/slam/{trans_xyz,quat_wxyz}",
        "gt_pose_convention": args.gt_pose_convention,
        "gt_camera_extrinsics_key": args.gt_camera_extrinsics_key,
        "gt_camera_extrinsics_convention": args.gt_camera_extrinsics_convention,
        "depth_source": "annotation.hdf5/depth/depth",
        "depth_scale": float(args.depth_scale),
        "sim3_alignment": sim3_alignment,
        "stats_vs_hdf5_slam_raw": raw_stats,
        "stats_vs_hdf5_slam_sim3_aligned": aligned_stats,
        "paths": paths,
    }
    json_path = output_dir / "comparison_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    text_path = output_dir / "comparison_report.txt"
    with text_path.open("w", encoding="utf-8") as file:
        file.write("Xperience-10M sample Mega-SAM report\n")
        file.write(f"scene_name={scene_name}\n")
        file.write(f"data_root={data_root}\n")
        file.write(f"frames={len(frame_ids)} source_range={frame_ids[0]}..{frame_ids[-1]} stride={args.stride}\n")
        file.write(f"fps={fps:.6f}\n")
        file.write(
            "coordinate=first-selected-frame-relative stereo_left camera trajectory; "
            f"HDF5 slam pose_convention={args.gt_pose_convention}, "
            f"extrinsics={args.gt_camera_extrinsics_key} "
            f"({args.gt_camera_extrinsics_convention})\n"
        )

        def write_stats(prefix: str, values: dict[str, float | int] | None) -> None:
            if values is None:
                file.write(f"{prefix}=None\n")
                return
            file.write(
                f"{prefix}="
                f"frames={values['frames']} "
                f"trans_rmse={values['translation_rmse']:.6f}m "
                f"trans_mean={values['translation_mean']:.6f}m "
                f"trans_max={values['translation_max']:.6f}m "
                f"rot_rmse={values['rotation_rmse']:.3f}deg "
                f"rot_mean={values['rotation_mean']:.3f}deg "
                f"rot_max={values['rotation_max']:.3f}deg\n"
            )

        write_stats("stats_vs_hdf5_slam_raw", raw_stats)
        write_stats("stats_vs_hdf5_slam_sim3_aligned", aligned_stats)
        if sim3_alignment.get("enabled"):
            file.write(
                "sim3_alignment="
                f"scale={float(sim3_alignment['scale']):.9g} "
                f"rotation_angle={float(sim3_alignment['rotation_angle_deg']):.3f}deg "
                f"translation={sim3_alignment['translation_xyz']} "
                f"used_samples={sim3_alignment['used_samples']} "
                f"residual_rmse={float(sim3_alignment['residual_rmse_m']):.6f}m\n"
            )
        else:
            file.write(f"sim3_alignment=disabled reason={sim3_alignment.get('reason', '')}\n")
        file.write("\n")
        for key, value in paths.items():
            file.write(f"{key}={value}\n")
    return text_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else sys.executable))
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--scene-name", default="")
    parser.add_argument("--video-name", default="stereo_left.mp4")
    parser.add_argument("--annotation-name", default="annotation.hdf5")
    parser.add_argument("--intrinsics-key", default="calibration/cam01/K")
    parser.add_argument(
        "--gt-pose-convention",
        choices=("world_from_camera", "camera_from_world"),
        default="camera_from_world",
        help="convention of annotation.hdf5 slam/{trans_xyz,quat_wxyz} before applying camera extrinsics",
    )
    parser.add_argument(
        "--gt-camera-extrinsics-key",
        default="calibration/cam01/T_c0_b",
        help="HDF5 4x4 extrinsic used to convert slam/body pose into the stereo_left camera pose; use 'none' to disable",
    )
    parser.add_argument(
        "--gt-camera-extrinsics-convention",
        choices=("camera_from_body", "body_from_camera"),
        default="camera_from_body",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=240,
        help="preview frame count; pass 0 to use every selected frame",
    )
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--skip-tracking", action="store_true")
    parser.add_argument("--keep-megasam-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--depth-min", type=float, default=0.05)
    parser.add_argument("--depth-max", type=float, default=10.0)
    parser.add_argument("--confidence-min", type=int, default=0)
    parser.add_argument("--no-sim3-align", action="store_true", help="skip writing Mega-SAM Sim3-aligned-to-GT trajectory")
    parser.add_argument("--alignment-min-displacement", type=float, default=0.0)
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--filter-thresh", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe-thresh", type=float, default=2.0)
    parser.add_argument("--frontend-thresh", type=float, default=12.0)
    parser.add_argument("--frontend-window", type=int, default=25)
    parser.add_argument("--frontend-radius", type=int, default=2)
    parser.add_argument("--frontend-nms", type=int, default=1)
    parser.add_argument("--backend-thresh", type=float, default=16.0)
    parser.add_argument("--backend-radius", type=int, default=2)
    parser.add_argument("--backend-nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument(
        "--megasam-mask-vis",
        action="store_true",
        help="save and log the valid mask passed into Mega-SAM; white keeps pixels, black excludes/downweights them",
    )
    parser.add_argument(
        "--megasam-motion-mask-vis",
        action="store_true",
        help="save and log Mega-SAM's learned motion mask; this keeps full BA enabled so the mask is produced",
    )
    parser.add_argument(
        "--enable-megasam-full-ba",
        action="store_true",
        help="run Mega-SAM full BA instead of passing --disable_full_ba",
    )
    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument("--save-rrd", type=Path)
    parser.add_argument("--rerun-spawn", action="store_true")
    parser.add_argument("--rerun-app-id", default="xperience_megasam")
    parser.add_argument("--max-rerun-frames", type=int)
    parser.add_argument("--rerun-trajectory-radius", type=float, default=0.01)
    parser.add_argument("--rerun-axis-length", type=float, default=0.06)
    parser.add_argument("--rerun-origin-radius", type=float, default=0.012)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    hdf5_path = args.data_root / args.annotation_name
    video_path = args.data_root / args.video_name
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing annotation HDF5: {hdf5_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Missing input video: {video_path}")
    if not args.weights.expanduser().exists():
        raise FileNotFoundError(f"Missing Mega-SAM checkpoint: {args.weights}")

    video_info = read_video_info(video_path)
    with h5py.File(hdf5_path, "r") as root:
        fps = xperience_fps(root, float(video_info["fps"]))
        hdf5_frames = int(root["depth/depth"].shape[0])
        total_frames = min(int(video_info["frames"]), hdf5_frames)
        max_frames = None if args.max_frames == 0 else args.max_frames
        frame_ids = select_frames(
            total_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            max_frames=max_frames,
        )
        timestamps = np.asarray(frame_ids, dtype=np.float64) / float(fps)
        intrinsics = calibration_matrix(root, args.intrinsics_key)
        fov_degrees = float(np.degrees(2.0 * np.arctan2(float(video_info["width"]), 2.0 * intrinsics[0, 0])))
        gt_world_from_camera = load_gt_world_from_camera(
            root,
            frame_ids,
            pose_convention=args.gt_pose_convention,
            camera_extrinsics_key=args.gt_camera_extrinsics_key,
            camera_extrinsics_convention=args.gt_camera_extrinsics_convention,
        )
        gt_rows = world_mats_to_relative_tum(gt_world_from_camera, timestamps)

        scene_name = sanitize_scene_name(args.scene_name or f"{args.data_root.name}_{Path(args.video_name).stem}")
        image_dir = args.output / "images"
        metric_root = args.output / "HDF5Depth" / "outputs"
        metric_scene_dir = metric_root / scene_name
        intrinsics_path = write_intrinsics_txt(args.output / "camera_intrinsic.txt", intrinsics)

        if args.skip_extract:
            if len(list(image_dir.glob("*.png"))) < len(frame_ids):
                raise FileNotFoundError(f"--skip-extract requested but {image_dir} is incomplete")
        else:
            print(f"Extracting {len(frame_ids)} frames from {video_path}")
            extract_frames(video_path, frame_ids, image_dir)

        if args.skip_depth:
            if len(list(metric_scene_dir.glob("*.npz"))) < len(frame_ids):
                raise FileNotFoundError(f"--skip-depth requested but {metric_scene_dir} is incomplete")
        else:
            print(f"Preparing {len(frame_ids)} metric depth maps from {hdf5_path}")
            prepare_metric_depth(
                root,
                frame_ids,
                metric_scene_dir,
                fov_degrees=fov_degrees,
                depth_scale=args.depth_scale,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                confidence_min=args.confidence_min,
            )

    np.savetxt(args.output / "frame_ids.txt", np.asarray(frame_ids, dtype=np.int64), fmt="%d")
    np.savetxt(args.output / "timestamps.txt", timestamps, fmt="%.6f")
    gt_tum_path = save_tum(args.output / "hdf5_slam_relative_tum.txt", gt_rows)

    if args.dry_run:
        run_megasam_tracking(
            args,
            image_dir=image_dir,
            metric_root=metric_root,
            intrinsics_path=intrinsics_path,
            scene_name=scene_name,
            frame_count=len(frame_ids),
        )
        print("dry-run complete; no Mega-SAM output converted.")
        return

    if args.skip_tracking:
        megasam_npz_path = args.output / "megasam_droid.npz"
        if not megasam_npz_path.exists():
            fallback = MEGASAM_ROOT / "outputs" / f"{scene_name}_droid.npz"
            if not fallback.exists():
                raise FileNotFoundError("Missing Mega-SAM npz for --skip-tracking")
            megasam_npz_path = fallback
    else:
        if not args.keep_megasam_cache:
            remove_megasam_scene_outputs(scene_name)
        megasam_npz_path = run_megasam_tracking(
            args,
            image_dir=image_dir,
            metric_root=metric_root,
            intrinsics_path=intrinsics_path,
            scene_name=scene_name,
            frame_count=len(frame_ids),
        )
        if not megasam_npz_path.exists():
            raise FileNotFoundError(f"Mega-SAM did not create expected output: {megasam_npz_path}")
        shutil.copy2(megasam_npz_path, args.output / "megasam_droid.npz")
        megasam_npz_path = args.output / "megasam_droid.npz"

    cam_c2w = load_megasam_cam_c2w(megasam_npz_path)
    count = min(len(cam_c2w), len(frame_ids), len(gt_rows))
    if count < len(frame_ids):
        print(f"WARNING: truncating trajectory to {count}/{len(frame_ids)} frames")
    megasam_rows = world_mats_to_relative_tum(cam_c2w[:count], timestamps[:count])
    megasam_tum_path = save_tum(args.output / "megasam_relative_tum.txt", megasam_rows)
    gt_rows = gt_rows[:count]
    frame_ids = frame_ids[:count]
    timestamps = timestamps[:count]
    gt_tum_path = save_tum(args.output / "hdf5_slam_relative_tum.txt", gt_rows)
    max_time_diff = 0.5 / float(fps) + 1e-6
    raw_stats = trajectory_stats(gt_rows, megasam_rows, max_time_diff=max_time_diff)

    aligned_rows = None
    aligned_stats = None
    aligned_tum_path = None
    if args.no_sim3_align:
        sim3_alignment: dict[str, Any] = {
            "enabled": False,
            "reason": "--no-sim3-align",
        }
    else:
        sim3_alignment = estimate_sim3_to_reference(
            gt_rows,
            megasam_rows,
            max_time_diff=max_time_diff,
            min_displacement=args.alignment_min_displacement,
        )
        aligned_rows = apply_sim3_to_rows(
            megasam_rows,
            scale=float(sim3_alignment["scale"]),
            rotation_matrix=sim3_alignment["rotation_matrix"],
            translation_xyz=sim3_alignment["translation_xyz"],
        )
        aligned_tum_path = save_tum(
            args.output / "sim3_aligned" / "megasam_sim3_aligned_tum.txt",
            aligned_rows,
        )
        sim3_alignment["aligned_tum"] = str(aligned_tum_path)
        sim3_alignment["source_tum"] = str(megasam_tum_path)
        sim3_alignment["reference_tum"] = str(gt_tum_path)
        aligned_stats = trajectory_stats(gt_rows, aligned_rows, max_time_diff=max_time_diff)

    save_rrd = None
    if not args.no_rerun:
        save_rrd = (args.save_rrd.expanduser().resolve() if args.save_rrd else args.output / "xperience_megasam.rrd")
        visualize_in_rerun(
            args,
            image_dir=image_dir,
            frame_ids=frame_ids,
            timestamps=timestamps,
            gt_rows=gt_rows,
            megasam_rows=megasam_rows,
            aligned_rows=aligned_rows,
            save_rrd=save_rrd,
        )

    paths = {
        "hdf5": str(hdf5_path),
        "video": str(video_path),
        "images": str(image_dir),
        "metric_depth": str(metric_scene_dir),
        "intrinsics": str(intrinsics_path),
        "gt_tum": str(gt_tum_path),
        "megasam_npz": str(megasam_npz_path),
        "megasam_tum": str(megasam_tum_path),
        "megasam_sim3_aligned_tum": str(aligned_tum_path) if aligned_tum_path is not None else "",
        "megasam_valid_mask_vis": str(args.output / "megasam_valid_mask_vis") if args.megasam_mask_vis else "",
        "megasam_motion_mask_vis": str(args.output / "megasam_motion_mask_vis") if args.megasam_motion_mask_vis else "",
        "rrd": str(save_rrd) if save_rrd is not None else "",
    }
    report_path = write_report(
        args.output,
        scene_name=scene_name,
        data_root=args.data_root,
        frame_ids=frame_ids,
        fps=fps,
        intrinsics=intrinsics,
        fov_degrees=fov_degrees,
        paths=paths,
        raw_stats=raw_stats,
        aligned_stats=aligned_stats,
        sim3_alignment=sim3_alignment,
        args=args,
    )

    print(f"Output directory: {args.output}")
    print(f"Saved HDF5 SLAM TUM: {gt_tum_path}")
    print(f"Saved Mega-SAM TUM: {megasam_tum_path}")
    if aligned_tum_path is not None:
        print(f"Saved Mega-SAM Sim3-aligned TUM: {aligned_tum_path}")
    if args.megasam_mask_vis:
        print(f"Saved Mega-SAM valid mask visualization: {args.output / 'megasam_valid_mask_vis'}")
    if args.megasam_motion_mask_vis:
        print(f"Saved Mega-SAM motion mask visualization: {args.output / 'megasam_motion_mask_vis'}")
    print(f"Saved report: {report_path}")
    if raw_stats is not None:
        print(
            "Raw stats vs HDF5 SLAM: "
            f"frames={raw_stats['frames']} "
            f"trans_rmse={raw_stats['translation_rmse']:.6f}m "
            f"rot_rmse={raw_stats['rotation_rmse']:.3f}deg"
        )
    if aligned_stats is not None:
        print(
            "Sim3-aligned stats vs HDF5 SLAM: "
            f"frames={aligned_stats['frames']} "
            f"trans_rmse={aligned_stats['translation_rmse']:.6f}m "
            f"rot_rmse={aligned_stats['rotation_rmse']:.3f}deg"
        )


if __name__ == "__main__":
    main()
