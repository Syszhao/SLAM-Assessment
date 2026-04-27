#!/usr/bin/env python3
"""Run Mega-SAM on an EgoDex hdf5/mp4 pair and visualize it with GT in Rerun.

This wrapper keeps Mega-SAM's original demo scripts intact. It prepares an
EgoDex image directory, runs the two depth preprocessors, runs
camera_tracking_scripts/test_demo.py, converts Mega-SAM's cam_c2w output to TUM,
and logs the estimate together with EgoDex GT in a first-frame-relative OpenCV
camera basis:

  x right, y down, z forward
"""

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
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import h5py
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing h5py. Install it in the Mega-SAM environment, e.g. "
        "`python -m pip install h5py`."
    ) from exc


MEGASAM_ROOT = Path(__file__).resolve().parent
DEFAULT_HDF5 = Path("/home/user/test/add_remove_lid/0.hdf5")
DEFAULT_OUTPUT_ROOT = MEGASAM_ROOT / "egodex_outputs"
DEFAULT_WEIGHTS = MEGASAM_ROOT / "checkpoints" / "megasam_final.pth"
DEFAULT_DEPTH_ANYTHING_CKPT = (
    MEGASAM_ROOT / "Depth-Anything" / "checkpoints" / "depth_anything_vitl14.pth"
)

OPENCV_FROM_ARKIT_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])
TUM_FMT = "%.6f %.9f %.9f %.9f %.9f %.9f %.9f %.9f"

GT_COLOR = [0, 200, 255]
MEGASAM_COLOR = [255, 90, 90]
MEGASAM_SCALE_COLOR = [255, 220, 0]
MEGASAM_SIM3_COLOR = [80, 220, 120]
AXIS_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]


def sanitize_scene_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "egodex_sequence"


def default_scene_name(hdf5_path: Path) -> str:
    return sanitize_scene_name(f"{hdf5_path.parent.name}_{hdf5_path.stem}")


def paired_mp4_path(hdf5_path: Path) -> Path:
    return hdf5_path.with_suffix(".mp4")


def resolve_path(path: Path, *, base: Path | None = None) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    if base is not None:
        base_candidate = (base / path).resolve()
        cwd_candidate = path.resolve()
        if base_candidate.exists() or not cwd_candidate.exists():
            return base_candidate
    return path.resolve()


def require_file(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


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


def python_command(args: argparse.Namespace, script: str, *script_args: str) -> list[str]:
    return [*shlex.split(args.python), script, *script_args]


def read_video_info(mp4_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {mp4_path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def pose_data_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape == (4, 4):
        return pose
    if pose.size == 7:
        pose_matrix = np.eye(4, dtype=np.float64)
        pose_matrix[:3, 3] = pose[:3]
        pose_matrix[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
        return pose_matrix
    raise ValueError(f"Cannot parse EgoDex pose shape: {pose.shape}")


def load_camera_poses(root: h5py.File) -> np.ndarray:
    camera_node = root["transforms"]["camera"]
    if isinstance(camera_node, h5py.Dataset):
        return camera_node[:].astype(np.float64)
    for key in ("matrix", "transform", "poses"):
        if key in camera_node:
            return camera_node[key][:].astype(np.float64)
    data_keys = [key for key in camera_node.keys() if "time" not in key.lower()]
    if not data_keys:
        raise ValueError("No usable pose dataset under transforms/camera")
    return camera_node[data_keys[0]][:].astype(np.float64)


def load_egodex_metadata(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as root:
        intrinsic = root["camera/intrinsic"][:].astype(np.float64)
        if intrinsic.shape != (3, 3):
            intrinsic = intrinsic.reshape(3, 3)
        pose_data = load_camera_poses(root)
        world_from_camera_arkit = np.stack([pose_data_to_matrix(pose) for pose in pose_data])
    return intrinsic, world_from_camera_arkit


def clear_files(directory: Path, patterns: tuple[str, ...]) -> None:
    if not directory.exists():
        return
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def extract_egodex_frames(
    mp4_path: Path,
    image_dir: Path,
    *,
    start_frame: int,
    end_frame: int | None,
    stride: int,
    max_frames: int | None,
) -> list[int]:
    image_dir.mkdir(parents=True, exist_ok=True)
    clear_files(image_dir, ("*.png", "*.jpg", "*.jpeg"))

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {mp4_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    source_frame_id = start_frame
    output_index = 0
    frame_ids: list[int] = []

    while True:
        if end_frame is not None and source_frame_id >= end_frame:
            break
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if (source_frame_id - start_frame) % stride == 0:
            output_path = image_dir / f"{output_index:06d}.png"
            if not cv2.imwrite(str(output_path), frame_bgr):
                raise IOError(f"Failed to write frame: {output_path}")
            frame_ids.append(source_frame_id)
            output_index += 1
            if max_frames is not None and output_index >= max_frames:
                break

        source_frame_id += 1

    cap.release()
    if not frame_ids:
        raise ValueError("No frames extracted; check --start-frame/--end-frame/--stride")
    return frame_ids


def load_frame_ids(path: Path) -> list[int]:
    rows = np.loadtxt(path, dtype=np.int64)
    if rows.ndim == 0:
        return [int(rows)]
    return [int(value) for value in rows.tolist()]


def save_tum(path: Path, rows: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(rows, dtype=np.float64), fmt=TUM_FMT)
    return path


def matrix_to_tum_row(timestamp: float, transform: np.ndarray) -> list[float]:
    quaternion = R.from_matrix(transform[:3, :3]).as_quat()
    quaternion /= np.linalg.norm(quaternion)
    return [float(timestamp), *transform[:3, 3].tolist(), *quaternion.tolist()]


def tum_row_to_matrix(row: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = row[1:4]
    transform[:3, :3] = R.from_quat(row[4:8]).as_matrix()
    return transform


def change_pose_basis(transform: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    return target_from_source @ transform @ np.linalg.inv(target_from_source)


def world_mats_to_relative_rows(
    world_from_camera: np.ndarray,
    frame_ids: list[int],
    fps: float,
    *,
    target_from_source: np.ndarray | None = None,
) -> np.ndarray:
    world_from_camera = np.asarray(world_from_camera, dtype=np.float64)
    if len(world_from_camera) != len(frame_ids):
        raise ValueError(
            f"Pose/frame count mismatch: {len(world_from_camera)} poses vs {len(frame_ids)} frames"
        )

    camera0_from_world = np.linalg.inv(world_from_camera[0])
    rows = []
    for frame_id, pose in zip(frame_ids, world_from_camera):
        camera0_from_camera = camera0_from_world @ pose
        if target_from_source is not None:
            camera0_from_camera = change_pose_basis(camera0_from_camera, target_from_source)
        rows.append(matrix_to_tum_row(frame_id / fps, camera0_from_camera))
    return np.asarray(rows, dtype=np.float64)


def is_valid_row(row: np.ndarray | None) -> bool:
    return row is not None and np.isfinite(np.asarray(row[1:8], dtype=np.float64)).all()


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    quaternions[valid] /= norms[valid]
    return quaternions


def match_by_timestamp(
    reference_rows: np.ndarray,
    estimate_rows: np.ndarray,
    max_time_diff: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference_rows = np.asarray(reference_rows, dtype=np.float64)
    estimate_rows = np.asarray(estimate_rows, dtype=np.float64)
    matched_ref = []
    matched_est = []
    ref_id = 0
    est_id = 0

    while ref_id < len(reference_rows) and est_id < len(estimate_rows):
        diff = reference_rows[ref_id, 0] - estimate_rows[est_id, 0]
        if abs(diff) <= max_time_diff:
            matched_ref.append(reference_rows[ref_id])
            matched_est.append(estimate_rows[est_id])
            ref_id += 1
            est_id += 1
        elif diff < 0:
            ref_id += 1
        else:
            est_id += 1

    return np.asarray(matched_ref, dtype=np.float64), np.asarray(matched_est, dtype=np.float64)


def rotation_error_deg(reference_rows: np.ndarray, estimate_rows: np.ndarray) -> np.ndarray:
    reference_rot = R.from_quat(normalize_quaternions(reference_rows[:, 4:8]))
    estimate_rot = R.from_quat(normalize_quaternions(estimate_rows[:, 4:8]))
    return np.degrees((estimate_rot.inv() * reference_rot).magnitude())


def trajectory_stats(
    reference_rows: np.ndarray,
    estimate_rows: np.ndarray | None,
    fps: float,
) -> dict[str, float | int] | None:
    if estimate_rows is None:
        return None
    ref, est = match_by_timestamp(reference_rows, estimate_rows, max_time_diff=0.5 / fps + 1e-6)
    if len(ref) == 0:
        return None
    finite = np.isfinite(ref[:, 1:8]).all(axis=1) & np.isfinite(est[:, 1:8]).all(axis=1)
    ref = ref[finite]
    est = est[finite]
    if len(ref) == 0:
        return None
    translation_errors = np.linalg.norm(est[:, 1:4] - ref[:, 1:4], axis=1)
    rotation_errors = rotation_error_deg(ref, est)
    return {
        "frames": int(len(ref)),
        "translation_rmse": float(np.sqrt(np.mean(translation_errors**2))),
        "translation_mean": float(np.mean(translation_errors)),
        "translation_max": float(np.max(translation_errors)),
        "rotation_rmse": float(np.sqrt(np.mean(rotation_errors**2))),
        "rotation_mean": float(np.mean(rotation_errors)),
        "rotation_max": float(np.max(rotation_errors)),
    }


def estimate_signed_lstsq_scale(
    reference_rows: np.ndarray,
    estimate_rows: np.ndarray,
    fps: float,
) -> float | None:
    """Least-squares scale that can become negative if directions disagree."""
    ref, est = match_by_timestamp(reference_rows, estimate_rows, max_time_diff=0.5 / fps + 1e-6)
    if len(ref) < 3:
        return None
    valid = (
        np.isfinite(ref[:, 1:4]).all(axis=1)
        & np.isfinite(est[:, 1:4]).all(axis=1)
        & (np.linalg.norm(est[:, 1:4], axis=1) > 1e-9)
    )
    ref_t = ref[valid, 1:4]
    est_t = est[valid, 1:4]
    denominator = float(np.sum(est_t * est_t))
    if denominator <= 1e-12:
        return None
    return float(np.sum(est_t * ref_t) / denominator)


def estimate_path_length_scale(
    reference_rows: np.ndarray,
    estimate_rows: np.ndarray,
    fps: float,
) -> float | None:
    """Positive scale based on accumulated camera-center motion length."""
    ref, est = match_by_timestamp(reference_rows, estimate_rows, max_time_diff=0.5 / fps + 1e-6)
    if len(ref) < 3:
        return None
    valid = np.isfinite(ref[:, 1:4]).all(axis=1) & np.isfinite(est[:, 1:4]).all(axis=1)
    ref_t = ref[valid, 1:4]
    est_t = est[valid, 1:4]
    if len(ref_t) < 3:
        return None
    ref_steps = np.linalg.norm(np.diff(ref_t, axis=0), axis=1)
    est_steps = np.linalg.norm(np.diff(est_t, axis=0), axis=1)
    ref_length = float(np.sum(ref_steps))
    est_length = float(np.sum(est_steps))
    if not np.isfinite(ref_length) or not np.isfinite(est_length) or est_length <= 1e-12:
        return None
    return ref_length / est_length


def choose_scale_only(
    method: str,
    *,
    signed_lstsq_scale: float | None,
    path_length_scale: float | None,
    sim3_similarity: dict[str, np.ndarray | float] | None,
) -> tuple[float | None, str]:
    if method == "umeyama":
        scale = float(sim3_similarity["scale"]) if sim3_similarity is not None else None
        if scale is not None:
            return scale, "umeyama_centered"
        if path_length_scale is not None:
            return path_length_scale, "path_length_ratio_fallback"
        if signed_lstsq_scale is not None:
            return signed_lstsq_scale, "signed_lstsq_fallback"
        return None, "umeyama_centered"

    if method == "path_length":
        if path_length_scale is not None:
            return path_length_scale, "path_length_ratio"
        if sim3_similarity is not None:
            return float(sim3_similarity["scale"]), "umeyama_centered_fallback"
        if signed_lstsq_scale is not None:
            return signed_lstsq_scale, "signed_lstsq_fallback"
        return None, "path_length_ratio"

    if method == "signed_lstsq":
        if signed_lstsq_scale is not None:
            return signed_lstsq_scale, "signed_lstsq"
        if sim3_similarity is not None:
            return float(sim3_similarity["scale"]), "umeyama_centered_fallback"
        if path_length_scale is not None:
            return path_length_scale, "path_length_ratio_fallback"
        return None, "signed_lstsq"

    raise ValueError(f"Unknown scale-only method: {method}")


def scale_tum_translations(rows: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.asarray(rows, dtype=np.float64).copy()
    scaled[:, 1:4] *= scale
    return scaled


def estimate_umeyama_similarity(
    reference_rows: np.ndarray,
    estimate_rows: np.ndarray,
    fps: float,
) -> dict[str, np.ndarray | float] | None:
    ref, est = match_by_timestamp(reference_rows, estimate_rows, max_time_diff=0.5 / fps + 1e-6)
    if len(ref) < 3:
        return None
    finite = np.isfinite(ref[:, 1:4]).all(axis=1) & np.isfinite(est[:, 1:4]).all(axis=1)
    ref_t = ref[finite, 1:4]
    est_t = est[finite, 1:4]
    if len(ref_t) < 3:
        return None

    est_mean = est_t.mean(axis=0)
    ref_mean = ref_t.mean(axis=0)
    est_centered = est_t - est_mean
    ref_centered = ref_t - ref_mean
    covariance = (ref_centered.T @ est_centered) / len(ref_t)
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u_mat) * np.linalg.det(vt_mat) < 0:
        sign[-1, -1] = -1.0
    rotation = u_mat @ sign @ vt_mat
    variance = float(np.sum(est_centered * est_centered) / len(est_t))
    if variance <= 1e-12:
        return None
    scale = float(np.trace(np.diag(singular_values) @ sign) / variance)
    if not np.isfinite(scale) or scale <= 0:
        return None
    translation = ref_mean - scale * (rotation @ est_mean)
    return {"scale": scale, "rotation": rotation, "translation": translation}


def apply_similarity_to_tum_rows(
    rows: np.ndarray,
    similarity: dict[str, np.ndarray | float],
) -> np.ndarray:
    scale = float(similarity["scale"])
    rotation = np.asarray(similarity["rotation"], dtype=np.float64)
    translation = np.asarray(similarity["translation"], dtype=np.float64)
    aligned = []
    for row in np.asarray(rows, dtype=np.float64):
        transform = tum_row_to_matrix(row)
        transform[:3, 3] = scale * (rotation @ transform[:3, 3]) + translation
        transform[:3, :3] = rotation @ transform[:3, :3]
        aligned.append(matrix_to_tum_row(row[0], transform))
    return np.asarray(aligned, dtype=np.float64)


def tum_rows_to_matrices(rows: np.ndarray) -> np.ndarray:
    return np.stack([tum_row_to_matrix(row) for row in np.asarray(rows, dtype=np.float64)])


def save_megasam_npz_with_rows(
    source_npz_path: Path,
    output_npz_path: Path,
    rows: np.ndarray,
    *,
    depth_scale: float | None,
) -> Path:
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source_npz_path) as data:
        payload = {key: data[key].copy() for key in data.files}

    payload["cam_c2w"] = tum_rows_to_matrices(rows)
    if depth_scale is not None and np.isfinite(depth_scale) and depth_scale > 0 and "depths" in payload:
        payload["depths"] = np.asarray(payload["depths"]) * float(depth_scale)
        payload["egodex_depth_scale_applied"] = np.asarray([float(depth_scale)], dtype=np.float64)
    np.savez(output_npz_path, **payload)
    return output_npz_path


def rows_by_frame_id(rows: np.ndarray | None, fps: float) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    if rows is None:
        return lookup
    for row in np.asarray(rows, dtype=np.float64):
        if is_valid_row(row):
            lookup[int(round(float(row[0]) * fps))] = row
    return lookup


def trajectory_points(rows: np.ndarray | None) -> np.ndarray | None:
    if rows is None:
        return None
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 8:
        return None
    valid = np.isfinite(rows[:, 1:8]).all(axis=1)
    points = rows[valid, 1:4]
    if len(points) < 2:
        return None
    return points


def log_static_trajectory(rr: Any, entity_path: str, rows: np.ndarray | None, color: list[int], radius: float) -> None:
    points = trajectory_points(rows)
    if points is None:
        return
    rr.log(
        f"{entity_path}/trajectory",
        rr.LineStrips3D(points, colors=[color], radii=radius),
        static=True,
    )


def log_pose(rr: Any, entity_path: str, row: np.ndarray | None, color: list[int], axis_length: float) -> None:
    if row is None or not is_valid_row(row):
        return
    quaternion = normalize_quaternions(np.asarray(row[4:8], dtype=np.float64)[None, :])[0]
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=np.asarray(row[1:4], dtype=np.float64),
            quaternion=quaternion,
        ),
    )
    rr.log(
        f"{entity_path}/axes",
        rr.Arrows3D(
            vectors=np.eye(3) * axis_length,
            colors=AXIS_COLORS,
            radii=axis_length * 0.05,
        ),
    )
    rr.log(
        f"{entity_path}/origin",
        rr.Points3D([[0.0, 0.0, 0.0]], colors=[color], radii=axis_length * 0.1),
    )


def rr_image(rr: Any, image_rgb: np.ndarray) -> Any:
    image = rr.Image(image_rgb)
    if hasattr(image, "compress"):
        return image.compress(jpeg_quality=80)
    return image


def set_rerun_time(rr: Any, frame_id: int, fps: float) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", frame_id)
        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds("time", frame_id / fps)
        return
    rr.set_time("frame", sequence=frame_id)
    rr.set_time("time", duration=frame_id / fps)


def init_rerun(args: argparse.Namespace) -> Any:
    try:
        import rerun as rr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing rerun-sdk. Install it in the Mega-SAM env, e.g. "
            "`python -m pip install rerun-sdk`, or pass --no-rerun."
        ) from exc

    save_rrd = Path(args.save_rrd) if args.save_rrd else None
    if save_rrd is not None:
        save_rrd.parent.mkdir(parents=True, exist_ok=True)

    rr.init(args.rerun_app_id, spawn=args.rerun_spawn and save_rrd is None)
    if save_rrd is not None:
        rr.save(str(save_rrd))

    try:
        view_coordinates = getattr(
            rr.ViewCoordinates,
            "RIGHT_HAND_Y_DOWN",
            rr.ViewCoordinates.RIGHT_HAND_Y_UP,
        )
        rr.log("world", view_coordinates, static=True)
    except Exception as exc:  # pragma: no cover - depends on rerun version
        print(f"Rerun coordinate setup failed, using defaults: {exc}")

    try:
        import rerun.blueprint as rrb

        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Horizontal(
                    column_shares=[0.42, 0.58],
                    contents=[
                        rrb.Spatial2DView(name="RGB", origin="world/input/image"),
                        rrb.Spatial3DView(name="GT vs Mega-SAM", origin="world"),
                    ],
                ),
            ),
            make_active=True,
        )
    except Exception as exc:  # pragma: no cover - depends on rerun version
        print(f"Rerun blueprint setup failed, using default layout: {exc}")

    return rr


def visualize_results_in_rerun(
    gt_rows: np.ndarray,
    megasam_rows: np.ndarray,
    megasam_scaled_rows: np.ndarray | None,
    megasam_sim3_rows: np.ndarray | None,
    mp4_path: Path,
    frame_ids: list[int],
    fps: float,
    args: argparse.Namespace,
) -> None:
    if args.no_rerun:
        return

    rr = init_rerun(args)
    log_static_trajectory(rr, "world/gt_opencv", gt_rows, GT_COLOR, args.rerun_trajectory_radius)
    log_static_trajectory(
        rr,
        "world/megasam_opencv",
        megasam_rows,
        MEGASAM_COLOR,
        args.rerun_trajectory_radius,
    )
    log_static_trajectory(
        rr,
        "world/megasam_opencv_scale_only",
        megasam_scaled_rows,
        MEGASAM_SCALE_COLOR,
        args.rerun_trajectory_radius,
    )
    log_static_trajectory(
        rr,
        "world/megasam_opencv_sim3",
        megasam_sim3_rows,
        MEGASAM_SIM3_COLOR,
        args.rerun_trajectory_radius,
    )

    gt_lookup = rows_by_frame_id(gt_rows, fps)
    megasam_lookup = rows_by_frame_id(megasam_rows, fps)
    scaled_lookup = rows_by_frame_id(megasam_scaled_rows, fps)
    sim3_lookup = rows_by_frame_id(megasam_sim3_rows, fps)
    selected = set(frame_ids)
    last_frame = max(selected)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"Cannot open video, Rerun will contain static trajectories only: {mp4_path}")
        return

    logged_frames = 0
    frame_id = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_id > last_frame:
            break
        if frame_id not in selected:
            frame_id += 1
            continue

        set_rerun_time(rr, frame_id, fps)
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rr.log("world/input/image", rr_image(rr, image_rgb))

        log_pose(rr, "world/gt_opencv/current", gt_lookup.get(frame_id), GT_COLOR, args.rerun_axis_length)
        log_pose(
            rr,
            "world/megasam_opencv/current",
            megasam_lookup.get(frame_id),
            MEGASAM_COLOR,
            args.rerun_axis_length,
        )
        log_pose(
            rr,
            "world/megasam_opencv_scale_only/current",
            scaled_lookup.get(frame_id),
            MEGASAM_SCALE_COLOR,
            args.rerun_axis_length,
        )
        sim3_row = sim3_lookup.get(frame_id)
        if sim3_row is not None and is_valid_row(sim3_row):
            rr.log(
                "world/megasam_opencv_sim3/current",
                rr.Points3D(
                    [np.asarray(sim3_row[1:4], dtype=np.float64)],
                    colors=[MEGASAM_SIM3_COLOR],
                    radii=args.rerun_axis_length * 0.1,
                ),
            )

        logged_frames += 1
        frame_id += 1

    cap.release()
    print(f"Sent Rerun visualization: {logged_frames} frames")
    if args.save_rrd:
        print(f"Saved Rerun recording: {args.save_rrd}")


def write_report(
    output_dir: Path,
    *,
    hdf5_path: Path,
    mp4_path: Path,
    scene_name: str,
    fps: float,
    frame_ids: list[int],
    megasam_pose_basis: str,
    scale: float | None,
    signed_lstsq_scale: float | None,
    path_length_scale: float | None,
    umeyama_scale: float | None,
    scale_method: str,
    sim3: dict[str, Any] | None,
    raw_stats: dict[str, float | int] | None,
    scaled_stats: dict[str, float | int] | None,
    sim3_stats: dict[str, float | int] | None,
    paths: dict[str, str],
) -> Path:
    report = {
        "scene_name": scene_name,
        "hdf5": str(hdf5_path),
        "mp4": str(mp4_path),
        "fps": fps,
        "frames": len(frame_ids),
        "first_frame": frame_ids[0],
        "last_frame": frame_ids[-1],
        "coordinate": "first-frame-relative OpenCV camera basis (x right, y down, z forward)",
        "megasam_pose_basis_assumption": megasam_pose_basis,
        "megasam_tum_default": (
            "megasam_relative_opencv_tum.txt is scale-only adjusted using scale_only_method "
            "when scale_only is available; the unscaled trajectory is saved as "
            "megasam_relative_opencv_raw_tum.txt. Aligned npz copies store cam_c2w in "
            "the same first-frame-relative OpenCV frame and scale depths when the "
            "alignment scale is positive."
        ),
        "scale_only": scale,
        "scale_only_method": scale_method,
        "signed_lstsq_scale_diagnostic": signed_lstsq_scale,
        "path_length_scale_diagnostic": path_length_scale,
        "umeyama_scale_diagnostic": umeyama_scale,
        "sim3": sim3,
        "raw_stats": raw_stats,
        "scaled_stats": scaled_stats,
        "sim3_stats": sim3_stats,
        "paths": paths,
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    text_path = output_dir / "comparison_report.txt"
    with text_path.open("w", encoding="utf-8") as file:
        file.write("EgoDex GT vs Mega-SAM\n")
        file.write("Coordinate: first-frame-relative OpenCV camera basis (x right, y down, z forward).\n")
        file.write(
            "Default Mega-SAM TUM: scale-only adjusted using scale_only_method when available; "
            "raw output is saved separately as megasam_relative_opencv_raw_tum.txt. "
            "Aligned npz copies use the same relative OpenCV frame and positive depth scale.\n"
        )
        file.write(f"scene_name={scene_name}\n")
        file.write(f"hdf5={hdf5_path}\n")
        file.write(f"mp4={mp4_path}\n")
        file.write(f"fps={fps:.6f}\n")
        file.write(f"frames={len(frame_ids)} frame {frame_ids[0]} -> {frame_ids[-1]}\n")
        file.write(f"megasam_pose_basis_assumption={megasam_pose_basis}\n")
        file.write(f"scale_only={scale}\n\n")
        file.write(f"scale_only_method={scale_method}\n")
        file.write(f"signed_lstsq_scale_diagnostic={signed_lstsq_scale}\n\n")
        file.write(f"path_length_scale_diagnostic={path_length_scale}\n")
        file.write(f"umeyama_scale_diagnostic={umeyama_scale}\n\n")
        if sim3 is not None:
            file.write(
                f"sim3_scale={sim3['scale']} "
                f"sim3_rotation_angle_deg={sim3['rotation_angle_deg']}\n\n"
            )
        for name, stats in (("raw", raw_stats), ("scale_only", scaled_stats), ("sim3", sim3_stats)):
            file.write(f"[{name}]\n")
            if stats is None:
                file.write("stats=None\n\n")
                continue
            if name == "sim3":
                file.write(
                    f"frames={stats['frames']} "
                    f"trans_rmse={stats['translation_rmse']:.6f}m "
                    f"trans_mean={stats['translation_mean']:.6f}m "
                    f"trans_max={stats['translation_max']:.6f}m "
                    "orientation_note=not_meaningful_after_similarity_alignment\n\n"
                )
                continue
            file.write(
                f"frames={stats['frames']} "
                f"trans_rmse={stats['translation_rmse']:.6f}m "
                f"trans_mean={stats['translation_mean']:.6f}m "
                f"trans_max={stats['translation_max']:.6f}m "
                f"rot_rmse={stats['rotation_rmse']:.3f}deg "
                f"rot_mean={stats['rotation_mean']:.3f}deg "
                f"rot_max={stats['rotation_max']:.3f}deg\n\n"
            )
        for key, value in paths.items():
            file.write(f"{key}={value}\n")
    return text_path


def maybe_run_evo(
    gt_tum_path: Path,
    estimate_tum_path: Path,
    output_dir: Path,
    fps: float,
    skip_evo: bool,
) -> Path | None:
    if skip_evo:
        return None
    try:
        from evo.core import metrics, sync
        from evo.tools import file_interface
    except ModuleNotFoundError as exc:
        print(f"Missing evo dependency; skipping evo metrics: {exc.name or 'evo'}")
        return None

    traj_ref = file_interface.read_tum_trajectory_file(str(gt_tum_path))
    traj_est = file_interface.read_tum_trajectory_file(str(estimate_tum_path))
    traj_ref, traj_est = sync.associate_trajectories(
        traj_ref,
        traj_est,
        max_diff=0.5 / fps + 1e-6,
    )
    evo_dir = output_dir / "evo_summary"
    evo_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"matched_poses": traj_ref.num_poses, "results": {}}

    relations = (
        ("position_m", metrics.PoseRelation.translation_part),
        ("orientation_deg", metrics.PoseRelation.rotation_angle_deg),
    )
    for name, relation in relations:
        ape = metrics.APE(relation)
        ape.process_data((traj_ref, traj_est))
        result = ape.get_result()
        results["results"][name] = {
            "title": result.info.get("title", name),
            "stats": {key: float(value) for key, value in result.stats.items()},
        }
        np.savetxt(evo_dir / f"ape_{name}_errors.txt", result.np_arrays["error_array"], fmt="%.9f")

    metrics_path = evo_dir / "evo_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved evo metrics: {metrics_path}")
    return metrics_path


def count_files(directory: Path, suffix: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(f"*{suffix}")))


def run_depth_preprocess(
    args: argparse.Namespace,
    *,
    image_dir: Path,
    scene_name: str,
    mono_root: Path,
    metric_root: Path,
) -> None:
    mono_scene = mono_root / scene_name
    metric_scene = metric_root / scene_name
    if mono_scene.exists():
        shutil.rmtree(mono_scene)
    if metric_scene.exists():
        shutil.rmtree(metric_scene)

    run_command(
        python_command(
            args,
            "Depth-Anything/run_videos.py",
            "--encoder",
            args.depth_anything_encoder,
            "--load-from",
            str(args.depth_anything_checkpoint),
            "--img-path",
            str(image_dir),
            "--outdir",
            str(mono_scene),
        ),
        cwd=MEGASAM_ROOT,
        dry_run=args.dry_run,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{MEGASAM_ROOT / 'UniDepth'}"
        if not env.get("PYTHONPATH")
        else f"{MEGASAM_ROOT / 'UniDepth'}:{env['PYTHONPATH']}"
    )
    run_command(
        python_command(
            args,
            "UniDepth/scripts/demo_mega-sam.py",
            "--scene-name",
            scene_name,
            "--img-path",
            str(image_dir),
            "--outdir",
            str(metric_root),
        ),
        cwd=MEGASAM_ROOT,
        env=env,
        dry_run=args.dry_run,
    )


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
    scene_name: str,
    mono_root: Path,
    metric_root: Path,
    intrinsic_path: Path,
) -> Path:
    if not args.dry_run and not args.keep_megasam_cache:
        remove_megasam_scene_outputs(scene_name)

    command = python_command(
        args,
        "camera_tracking_scripts/test_demo.py",
        "--datapath",
        str(image_dir),
        "--weights",
        str(args.weights),
        "--scene_name",
        scene_name,
        "--mono_depth_path",
        str(mono_root),
        "--metric_depth_path",
        str(metric_root),
        "--intrinsics",
        str(intrinsic_path),
        "--disable_vis",
        "--buffer",
        str(args.buffer),
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
    )
    if args.upsample:
        command.append("--upsample")
    if args.opt_focal:
        command.append("--opt_focal")

    run_command(command, cwd=MEGASAM_ROOT, dry_run=args.dry_run)
    return MEGASAM_ROOT / "outputs" / f"{scene_name}_droid.npz"


def load_megasam_cam_c2w(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"Mega-SAM output not found: {npz_path}")
    data = np.load(npz_path)
    if "cam_c2w" not in data:
        raise KeyError(f"{npz_path} does not contain cam_c2w")
    cam_c2w = np.asarray(data["cam_c2w"], dtype=np.float64)
    if cam_c2w.ndim != 3 or cam_c2w.shape[1:] != (4, 4):
        raise ValueError(f"cam_c2w must have shape Nx4x4, got {cam_c2w.shape}")
    return cam_c2w


def existing_megasam_npz(output_dir: Path, scene_name: str) -> Path:
    local_copy = output_dir / "megasam_droid.npz"
    if local_copy.exists():
        return local_copy
    return MEGASAM_ROOT / "outputs" / f"{scene_name}_droid.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mega-SAM on an EgoDex sequence and visualize estimate + GT in Rerun."
    )
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5, help="EgoDex .hdf5 path")
    parser.add_argument("--mp4", type=Path, help="EgoDex .mp4 path; default is <hdf5 stem>.mp4")
    parser.add_argument("--output", type=Path, help="output directory; default is mega-sam/egodex_outputs/<scene>")
    parser.add_argument("--scene-name", help="Mega-SAM scene name; default is <task>_<sequence>")
    parser.add_argument("--fps", type=float, help="override video FPS; default reads FPS from mp4")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="exclusive end frame")
    parser.add_argument("--max-frames", type=int, help="debug limit after frame filtering")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth video frame")
    parser.add_argument("--skip-extract", action="store_true", help="reuse output images and frame_ids.txt")
    parser.add_argument("--skip-depth", action="store_true", help="reuse Depth-Anything and UniDepth outputs")
    parser.add_argument("--skip-tracking", action="store_true", help="reuse existing Mega-SAM output npz")
    parser.add_argument("--prepare-only", action="store_true", help="stop after extracting frames and GT TUM")
    parser.add_argument("--dry-run", action="store_true", help="print external commands without running them")

    parser.add_argument(
        "--python",
        default=sys.executable,
        help='Python command for Mega-SAM subprocesses, e.g. "conda run -n mega_sam python"',
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--depth-anything-checkpoint", type=Path, default=DEFAULT_DEPTH_ANYTHING_CKPT)
    parser.add_argument("--depth-anything-encoder", choices=("vits", "vitb", "vitl"), default="vitl")
    parser.add_argument("--keep-megasam-cache", action="store_true", help="do not remove Mega-SAM outputs before tracking")

    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=2.0)
    parser.add_argument("--frontend_thresh", type=float, default=12.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=16.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--opt-focal", action="store_true", help="enable Mega-SAM focal optimization")

    parser.add_argument(
        "--megasam-pose-basis",
        choices=("opencv", "arkit"),
        default="opencv",
        help="basis assumption for Mega-SAM cam_c2w; use arkit to apply a y/z flip before comparison",
    )
    parser.add_argument(
        "--scale-only-method",
        choices=("umeyama", "path_length", "signed_lstsq"),
        default="umeyama",
        help=(
            "scale used for megasam_relative_opencv_scale_only_tum.txt. "
            "umeyama is the standard centered Sim(3) scale; path_length matches GT total motion length; "
            "signed_lstsq can become negative when translation directions disagree."
        ),
    )
    parser.add_argument("--skip-evo", action="store_true", help="skip evo APE metrics")

    parser.add_argument("--no-rerun", action="store_true", help="skip Rerun visualization")
    parser.add_argument("--rerun-app-id", default="egodex_megasam")
    parser.add_argument("--rerun-spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rrd", help="save a .rrd recording instead of spawning the viewer")
    parser.add_argument("--rerun-axis-length", type=float, default=0.04)
    parser.add_argument("--rerun-trajectory-radius", type=float, default=0.002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0")

    hdf5_path = args.hdf5.expanduser().resolve()
    mp4_path = (args.mp4 or paired_mp4_path(hdf5_path)).expanduser().resolve()
    args.weights = resolve_path(args.weights, base=MEGASAM_ROOT)
    args.depth_anything_checkpoint = resolve_path(args.depth_anything_checkpoint, base=MEGASAM_ROOT)
    scene_name = sanitize_scene_name(args.scene_name) if args.scene_name else default_scene_name(hdf5_path)
    output_dir = (args.output or (DEFAULT_OUTPUT_ROOT / scene_name)).expanduser().resolve()
    image_dir = output_dir / "images"
    mono_root = output_dir / "Depth-Anything" / "video_visualization"
    metric_root = output_dir / "UniDepth" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    require_file(hdf5_path, "EgoDex hdf5")
    require_file(mp4_path, "EgoDex mp4")
    if not args.skip_depth and not args.prepare_only and not args.dry_run:
        require_file(args.depth_anything_checkpoint, "Depth-Anything checkpoint")
    if not args.skip_tracking and not args.prepare_only and not args.dry_run:
        require_file(args.weights, "Mega-SAM checkpoint")

    intrinsic, world_from_camera_arkit = load_egodex_metadata(hdf5_path)
    video_info = read_video_info(mp4_path)
    fps = args.fps or video_info["fps"] or 30.0
    np.savetxt(output_dir / "camera_intrinsic.txt", intrinsic)

    print(f"Output directory: {output_dir}")
    print(
        f"EgoDex video: {video_info['frames']} frames, "
        f"{video_info['width']}x{video_info['height']}, fps={fps:.3f}"
    )
    print(f"EgoDex GT: {len(world_from_camera_arkit)} poses")
    print(f"Mega-SAM scene: {scene_name}")

    frame_ids_path = output_dir / "frame_ids.txt"
    if args.skip_extract:
        require_file(frame_ids_path, "frame_ids.txt for --skip-extract")
        frame_ids = load_frame_ids(frame_ids_path)
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory for --skip-extract: {image_dir}")
    else:
        frame_ids = extract_egodex_frames(
            mp4_path,
            image_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            max_frames=args.max_frames,
        )
        np.savetxt(frame_ids_path, np.asarray(frame_ids, dtype=np.int64), fmt="%d")

    if max(frame_ids) >= len(world_from_camera_arkit):
        raise ValueError(f"Selected frame {max(frame_ids)} exceeds GT length {len(world_from_camera_arkit)}")
    print(f"Mega-SAM input: {len(frame_ids)} frames, frame {frame_ids[0]} -> {frame_ids[-1]}")

    gt_world_selected = world_from_camera_arkit[np.asarray(frame_ids, dtype=np.int64)]
    gt_rows = world_mats_to_relative_rows(
        gt_world_selected,
        frame_ids,
        fps,
        target_from_source=OPENCV_FROM_ARKIT_CAMERA,
    )
    gt_tum_path = save_tum(output_dir / "gt_relative_opencv_tum.txt", gt_rows)
    print(f"Saved GT TUM: {gt_tum_path}")

    if args.prepare_only:
        print("prepare-only complete; depth/tracking/Rerun were not run.")
        return

    if not args.skip_depth:
        run_depth_preprocess(
            args,
            image_dir=image_dir,
            scene_name=scene_name,
            mono_root=mono_root,
            metric_root=metric_root,
        )

    if args.dry_run:
        if not args.skip_tracking:
            run_megasam_tracking(
                args,
                image_dir=image_dir,
                scene_name=scene_name,
                mono_root=mono_root,
                metric_root=metric_root,
                intrinsic_path=output_dir / "camera_intrinsic.txt",
            )
        print("dry-run complete; conversion/Rerun were not run.")
        return

    mono_count = count_files(mono_root / scene_name, ".npy")
    metric_count = count_files(metric_root / scene_name, ".npz")
    if mono_count < len(frame_ids) or metric_count < len(frame_ids):
        raise RuntimeError(
            "Depth preprocessing outputs are incomplete: "
            f"Depth-Anything {mono_count}/{len(frame_ids)}, "
            f"UniDepth {metric_count}/{len(frame_ids)}"
        )

    if args.skip_tracking:
        megasam_npz_path = existing_megasam_npz(output_dir, scene_name)
    else:
        megasam_npz_path = run_megasam_tracking(
            args,
            image_dir=image_dir,
            scene_name=scene_name,
            mono_root=mono_root,
            metric_root=metric_root,
            intrinsic_path=output_dir / "camera_intrinsic.txt",
        )
        if not megasam_npz_path.exists():
            raise FileNotFoundError(f"Mega-SAM did not create expected output: {megasam_npz_path}")
        shutil.copy2(megasam_npz_path, output_dir / "megasam_droid.npz")
        reconstruction_dir = MEGASAM_ROOT / "reconstructions" / scene_name
        if reconstruction_dir.exists():
            local_reconstruction = output_dir / "reconstruction"
            if local_reconstruction.exists():
                shutil.rmtree(local_reconstruction)
            shutil.copytree(reconstruction_dir, local_reconstruction)
        megasam_npz_path = output_dir / "megasam_droid.npz"

    cam_c2w = load_megasam_cam_c2w(megasam_npz_path)
    if len(cam_c2w) < len(frame_ids):
        print(f"Mega-SAM returned {len(cam_c2w)} poses; truncating GT/frame ids to match.")
        frame_ids = frame_ids[: len(cam_c2w)]
        gt_rows = gt_rows[: len(cam_c2w)]
        gt_tum_path = save_tum(output_dir / "gt_relative_opencv_tum.txt", gt_rows)
    elif len(cam_c2w) > len(frame_ids):
        cam_c2w = cam_c2w[: len(frame_ids)]

    target_from_source = OPENCV_FROM_ARKIT_CAMERA if args.megasam_pose_basis == "arkit" else None
    megasam_raw_rows = world_mats_to_relative_rows(
        cam_c2w,
        frame_ids,
        fps,
        target_from_source=target_from_source,
    )
    megasam_raw_tum_path = save_tum(
        output_dir / "megasam_relative_opencv_raw_tum.txt",
        megasam_raw_rows,
    )

    signed_lstsq_scale = estimate_signed_lstsq_scale(gt_rows, megasam_raw_rows, fps)
    path_length_scale = estimate_path_length_scale(gt_rows, megasam_raw_rows, fps)
    sim3_similarity = estimate_umeyama_similarity(gt_rows, megasam_raw_rows, fps)
    umeyama_scale = float(sim3_similarity["scale"]) if sim3_similarity is not None else None
    scale, scale_method = choose_scale_only(
        args.scale_only_method,
        signed_lstsq_scale=signed_lstsq_scale,
        path_length_scale=path_length_scale,
        sim3_similarity=sim3_similarity,
    )
    megasam_rows = megasam_raw_rows
    megasam_scaled_rows = None
    megasam_scaled_tum_path = None
    megasam_scaled_npz_path = None
    if scale is not None:
        megasam_scaled_rows = scale_tum_translations(megasam_raw_rows, scale)
        megasam_scaled_tum_path = save_tum(
            output_dir / "megasam_relative_opencv_scale_only_tum.txt",
            megasam_scaled_rows,
        )
        megasam_scaled_npz_path = save_megasam_npz_with_rows(
            megasam_npz_path,
            output_dir / "megasam_relative_opencv_scale_only_droid.npz",
            megasam_scaled_rows,
            depth_scale=scale,
        )
        megasam_rows = megasam_scaled_rows
        print(f"scale_only = {scale:.9f} ({scale_method})")
        if args.scale_only_method != "signed_lstsq" and signed_lstsq_scale is not None and signed_lstsq_scale < 0:
            print(
                "signed_lstsq_scale is negative "
                f"({signed_lstsq_scale:.9f}); translation direction differs from GT, "
                "so positive scale-only will not fully overlay the trajectories."
            )
    else:
        print("Could not estimate scale_only; only raw Mega-SAM trajectory will be logged.")
    megasam_tum_path = save_tum(output_dir / "megasam_relative_opencv_tum.txt", megasam_rows)

    raw_stats = trajectory_stats(gt_rows, megasam_raw_rows, fps)
    scaled_stats = trajectory_stats(gt_rows, megasam_scaled_rows, fps)
    megasam_sim3_rows = None
    megasam_sim3_tum_path = None
    megasam_sim3_npz_path = None
    sim3_report = None
    sim3_stats = None
    if sim3_similarity is not None:
        megasam_sim3_rows = apply_similarity_to_tum_rows(megasam_raw_rows, sim3_similarity)
        megasam_sim3_tum_path = save_tum(
            output_dir / "megasam_relative_opencv_sim3_tum.txt",
            megasam_sim3_rows,
        )
        megasam_sim3_npz_path = save_megasam_npz_with_rows(
            megasam_npz_path,
            output_dir / "megasam_relative_opencv_sim3_droid.npz",
            megasam_sim3_rows,
            depth_scale=float(sim3_similarity["scale"]),
        )
        rotation = np.asarray(sim3_similarity["rotation"], dtype=np.float64)
        sim3_report = {
            "scale": float(sim3_similarity["scale"]),
            "rotation_angle_deg": float(np.degrees(R.from_matrix(rotation).magnitude())),
            "translation": np.asarray(sim3_similarity["translation"], dtype=np.float64).tolist(),
        }
        sim3_stats = trajectory_stats(gt_rows, megasam_sim3_rows, fps)

    paths = {
        "frame_ids": str(frame_ids_path),
        "gt_tum": str(gt_tum_path),
        "megasam_npz": str(megasam_npz_path),
        "megasam_tum": str(megasam_tum_path),
        "megasam_raw_tum": str(megasam_raw_tum_path),
    }
    if megasam_scaled_tum_path is not None:
        paths["megasam_scale_only_tum"] = str(megasam_scaled_tum_path)
    if megasam_scaled_npz_path is not None:
        paths["megasam_scale_only_npz"] = str(megasam_scaled_npz_path)
    if megasam_sim3_tum_path is not None:
        paths["megasam_sim3_tum"] = str(megasam_sim3_tum_path)
    if megasam_sim3_npz_path is not None:
        paths["megasam_sim3_npz"] = str(megasam_sim3_npz_path)

    report_path = write_report(
        output_dir,
        hdf5_path=hdf5_path,
        mp4_path=mp4_path,
        scene_name=scene_name,
        fps=fps,
        frame_ids=frame_ids,
        megasam_pose_basis=args.megasam_pose_basis,
        scale=scale,
        signed_lstsq_scale=signed_lstsq_scale,
        path_length_scale=path_length_scale,
        umeyama_scale=umeyama_scale,
        scale_method=scale_method,
        sim3=sim3_report,
        raw_stats=raw_stats,
        scaled_stats=scaled_stats,
        sim3_stats=sim3_stats,
        paths=paths,
    )
    print(f"Saved comparison report: {report_path}")
    if raw_stats is not None:
        print(
            "Raw Mega-SAM | "
            f"translation RMSE {raw_stats['translation_rmse']:.6f} m | "
            f"rotation RMSE {raw_stats['rotation_rmse']:.3f} deg"
        )
    if scaled_stats is not None:
        print(
            "Scale-only Mega-SAM | "
            f"translation RMSE {scaled_stats['translation_rmse']:.6f} m | "
            f"rotation RMSE {scaled_stats['rotation_rmse']:.3f} deg"
        )
    if sim3_stats is not None:
        print(
            "Sim3-aligned Mega-SAM | "
            f"translation RMSE {sim3_stats['translation_rmse']:.6f} m "
            "(trajectory reference only)"
        )

    if megasam_scaled_tum_path is not None:
        maybe_run_evo(gt_tum_path, megasam_scaled_tum_path, output_dir, fps, args.skip_evo)

    visualize_results_in_rerun(
        gt_rows,
        megasam_raw_rows,
        megasam_scaled_rows,
        megasam_sim3_rows,
        mp4_path,
        frame_ids,
        fps,
        args,
    )


if __name__ == "__main__":
    main()
