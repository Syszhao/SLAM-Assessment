#!/usr/bin/env python3
"""Shared helpers for LeRobot v3 stereo SLAM wrappers."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterator

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_DATASET_ROOT = Path.home() / "test_data"
DEFAULT_LEFT_KEY = "observation.images.head_stereo_left_rec"
DEFAULT_RIGHT_KEY = "observation.images.head_stereo_right_rec"
DEFAULT_CAMERA_PARAMS_KEY = "head_stereo"
DEFAULT_GT_COLUMN = "auto"
TUM_FMT = "%.6f %.9f %.9f %.9f %.9f %.9f %.9f %.9f"

# LeRobot/robot-base style: x forward, y left, z up.
OPENCV_FROM_ROBOT_BASE = np.eye(4, dtype=np.float64)
OPENCV_FROM_ROBOT_BASE[:3, :3] = np.asarray(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# MAC-VO reports poses in its NED camera convention: x forward, y right, z down.
OPENCV_FROM_MACVO_NED_CAMERA = np.eye(4, dtype=np.float64)
OPENCV_FROM_MACVO_NED_CAMERA[:3, :3] = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

AXIS_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]


def require_file(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def require_dir(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_tum(path: Path, rows: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(rows, dtype=np.float64), fmt=TUM_FMT)
    return path


def load_tum(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise ValueError(f"TUM file must have shape Nx8: {path}")
    rows = rows[np.isfinite(rows[:, 1:8]).all(axis=1)]
    rows[:, 4:8] = normalize_quaternions(rows[:, 4:8])
    return rows


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    quaternions[valid] /= norms[valid]
    return quaternions


def matrix_to_tum_row(timestamp: float, transform: np.ndarray) -> list[float]:
    quaternion = R.from_matrix(transform[:3, :3]).as_quat()
    quaternion /= np.linalg.norm(quaternion)
    return [float(timestamp), *transform[:3, 3].tolist(), *quaternion.tolist()]


def tum_row_to_matrix(row: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(row[1:4], dtype=np.float64)
    transform[:3, :3] = R.from_quat(np.asarray(row[4:8], dtype=np.float64)).as_matrix()
    return transform


def se3_row_to_matrix(row: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(row[:3], dtype=np.float64)
    transform[:3, :3] = R.from_quat(np.asarray(row[3:7], dtype=np.float64)).as_matrix()
    return transform


def change_pose_basis(transform: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    return target_from_source @ transform @ np.linalg.inv(target_from_source)


def pose6_to_matrix(pose: np.ndarray, euler_order: str) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Expected pose vector [x y z roll pitch yaw], got {pose.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = pose[:3]
    transform[:3, :3] = R.from_euler(euler_order, pose[3:6]).as_matrix()
    return transform


def target_from_source_frame(source_frame: str, target_frame: str) -> np.ndarray:
    if source_frame == target_frame:
        return np.eye(4, dtype=np.float64)
    if source_frame == "robot_base" and target_frame == "opencv":
        return OPENCV_FROM_ROBOT_BASE.copy()
    if source_frame == "opencv" and target_frame == "robot_base":
        return np.linalg.inv(OPENCV_FROM_ROBOT_BASE)
    raise ValueError(f"Unsupported coordinate conversion: {source_frame} -> {target_frame}")


def pose6_rows_to_relative_tum(
    poses: np.ndarray,
    timestamps: np.ndarray,
    euler_order: str,
    *,
    source_frame: str,
    target_frame: str,
    pose_convention: str,
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(poses) != len(timestamps):
        raise ValueError(f"GT pose/timestamp count mismatch: {len(poses)} vs {len(timestamps)}")

    matrices = np.stack([pose6_to_matrix(row, euler_order) for row in poses])
    if pose_convention == "camera_from_world":
        matrices = np.linalg.inv(matrices)
    elif pose_convention != "world_from_camera":
        raise ValueError(f"Unsupported pose convention: {pose_convention}")

    origin_from_world = np.linalg.inv(matrices[0])
    target_from_source = target_from_source_frame(source_frame, target_frame)
    rows = [
        matrix_to_tum_row(
            timestamp,
            change_pose_basis(origin_from_world @ matrix, target_from_source),
        )
        for timestamp, matrix in zip(timestamps, matrices)
    ]
    return np.asarray(rows, dtype=np.float64)


def world_mats_to_relative_tum(
    world_from_camera: np.ndarray,
    timestamps: np.ndarray,
    *,
    target_from_source: np.ndarray | None = None,
) -> np.ndarray:
    world_from_camera = np.asarray(world_from_camera, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(world_from_camera) != len(timestamps):
        raise ValueError(
            f"Pose/timestamp count mismatch: {len(world_from_camera)} vs {len(timestamps)}"
        )

    camera0_from_world = np.linalg.inv(world_from_camera[0])
    rows = []
    for timestamp, pose in zip(timestamps, world_from_camera):
        camera0_from_camera = camera0_from_world @ pose
        if target_from_source is not None:
            camera0_from_camera = change_pose_basis(camera0_from_camera, target_from_source)
        rows.append(matrix_to_tum_row(float(timestamp), camera0_from_camera))
    return np.asarray(rows, dtype=np.float64)


def time_key(timestamp: float) -> int:
    return int(round(float(timestamp) * 1_000_000.0))


def rows_by_time_key(rows: np.ndarray | None) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    if rows is None:
        return lookup
    for row in np.asarray(rows, dtype=np.float64):
        if np.isfinite(row[1:8]).all():
            lookup[time_key(row[0])] = row
    return lookup


def match_by_timestamp(
    reference_rows: np.ndarray | None,
    estimate_rows: np.ndarray | None,
    max_time_diff: float,
) -> tuple[np.ndarray, np.ndarray]:
    if reference_rows is None or estimate_rows is None:
        return np.empty((0, 8), dtype=np.float64), np.empty((0, 8), dtype=np.float64)

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
    reference_rows: np.ndarray | None,
    estimate_rows: np.ndarray | None,
    max_time_diff: float,
    *,
    include_rotation: bool = True,
) -> dict[str, float | int] | None:
    ref, est = match_by_timestamp(reference_rows, estimate_rows, max_time_diff)
    if len(ref) == 0:
        return None
    finite = np.isfinite(ref[:, 1:8]).all(axis=1) & np.isfinite(est[:, 1:8]).all(axis=1)
    ref = ref[finite]
    est = est[finite]
    if len(ref) == 0:
        return None
    translation_errors = np.linalg.norm(est[:, 1:4] - ref[:, 1:4], axis=1)
    result: dict[str, float | int] = {
        "frames": int(len(ref)),
        "translation_rmse": float(np.sqrt(np.mean(translation_errors**2))),
        "translation_mean": float(np.mean(translation_errors)),
        "translation_max": float(np.max(translation_errors)),
    }
    if include_rotation:
        rotation_errors = rotation_error_deg(ref, est)
        result.update(
            rotation_rmse=float(np.sqrt(np.mean(rotation_errors**2))),
            rotation_mean=float(np.mean(rotation_errors)),
            rotation_max=float(np.max(rotation_errors)),
        )
    return result


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


def rr_image(rr: Any, image_rgb: np.ndarray) -> Any:
    image = rr.Image(np.ascontiguousarray(image_rgb))
    if hasattr(image, "compress"):
        return image.compress(jpeg_quality=80)
    return image


def set_rerun_time(rr: Any, frame_id: int, timestamp: float) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", int(frame_id))
        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds("time", float(timestamp))
        return
    rr.set_time("frame", sequence=int(frame_id))
    rr.set_time("time", duration=float(timestamp))


def log_static_trajectory(rr: Any, entity_path: str, rows: np.ndarray | None, color: list[int], radius: float) -> None:
    points = trajectory_points(rows)
    if points is None:
        return
    rr.log(
        f"{entity_path}/trajectory",
        rr.LineStrips3D(points, colors=[color], radii=radius),
        static=True,
    )


def log_pose(
    rr: Any,
    entity_path: str,
    row: np.ndarray | None,
    color: list[int],
    axis_length: float,
    origin_radius: float,
) -> None:
    if row is None or not np.isfinite(np.asarray(row[1:8], dtype=np.float64)).all():
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
        rr.Points3D([[0.0, 0.0, 0.0]], colors=[color], radii=origin_radius),
    )


def feature_fps(info: dict[str, Any], key: str) -> float | None:
    feature = info.get("features", {}).get(key, {})
    video_info = feature.get("info", {})
    fps = video_info.get("video.fps")
    return float(fps) if fps else None


def dataset_fps(info: dict[str, Any], left_key: str) -> float:
    return float(feature_fps(info, left_key) or info.get("fps") or 30.0)


def camera_params(info: dict[str, Any], key: str) -> dict[str, Any]:
    params = info.get("camera_params", {}).get(key)
    if not params:
        raise KeyError(f"Missing camera_params.{key} in meta/info.json")
    return params


def pose6_feature_columns(info: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for key, feature in info.get("features", {}).items():
        shape = feature.get("shape")
        names = [str(name).lower() for name in feature.get("names") or []]
        is_pose6 = list(shape or []) == [6] and names[:6] == ["x", "y", "z", "roll", "pitch", "yaw"]
        if is_pose6:
            columns.append(key)
    return columns


def resolve_gt_column(info: dict[str, Any], requested: str, camera_params_key: str) -> str:
    if requested != "auto":
        if requested not in info.get("features", {}):
            candidates = ", ".join(pose6_feature_columns(info)) or "<none>"
            raise KeyError(
                f"GT column {requested!r} is not in meta/info.json features. Pose6 candidates: {candidates}"
            )
        return requested

    candidates = pose6_feature_columns(info)
    if not candidates:
        raise KeyError("No [x y z roll pitch yaw] pose feature found in meta/info.json")

    tokens = [token for token in camera_params_key.lower().split("_") if token]

    def score(column: str) -> tuple[int, str]:
        lower = column.lower()
        value = 0
        if lower.startswith("observation.state."):
            value += 20
        if "state_d435" in lower:
            value += 100
        if "d435" in lower:
            value += 60
        if "camera" in lower:
            value += 40
        if all(token in lower for token in tokens):
            value += 30
        if "torso" in lower:
            value += 5
        return value, column

    return max(candidates, key=score)


def import_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing pyarrow. Install it in the environment running this script, "
            "for example `python -m pip install pyarrow`."
        ) from exc
    return pc, ds


def video_path(dataset_root: Path, key: str, chunk_index: int, file_index: int) -> Path:
    return (
        dataset_root
        / "videos"
        / key
        / f"chunk-{int(chunk_index):03d}"
        / f"file-{int(file_index):03d}.mp4"
    )


def load_episode_info(
    dataset_root: Path,
    episode_index: int,
    left_key: str,
    right_key: str,
    fps: float,
) -> dict[str, Any]:
    pc, ds = import_pyarrow()
    episode_root = dataset_root / "meta" / "episodes"
    require_dir(episode_root, "LeRobot episode metadata")

    def segment_columns(key: str) -> list[str]:
        prefix = f"videos/{key}"
        return [
            f"{prefix}/chunk_index",
            f"{prefix}/file_index",
            f"{prefix}/from_timestamp",
            f"{prefix}/to_timestamp",
        ]

    columns = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        *segment_columns(left_key),
        *segment_columns(right_key),
    ]
    dataset = ds.dataset(str(episode_root), format="parquet")
    schema_names = set(dataset.schema.names)
    missing = [column for column in columns if column not in schema_names]
    if missing:
        raise KeyError("Episode metadata is missing required columns: " + ", ".join(missing))

    table = dataset.to_table(
        columns=columns,
        filter=pc.field("episode_index") == int(episode_index),
    )
    if table.num_rows != 1:
        raise ValueError(f"Expected one metadata row for episode {episode_index}, got {table.num_rows}")
    row = {key: values[0] for key, values in table.to_pydict().items()}

    def make_segment(key: str) -> dict[str, Any]:
        prefix = f"videos/{key}"
        path = video_path(
            dataset_root,
            key,
            int(row[f"{prefix}/chunk_index"]),
            int(row[f"{prefix}/file_index"]),
        )
        require_file(path, f"{key} video")
        from_timestamp = float(row[f"{prefix}/from_timestamp"])
        to_timestamp = float(row[f"{prefix}/to_timestamp"])
        return {
            "key": key,
            "path": str(path.resolve()),
            "from_timestamp": from_timestamp,
            "to_timestamp": to_timestamp,
            "start_frame": int(round(from_timestamp * fps)),
            "duration_frames": int(round((to_timestamp - from_timestamp) * fps)),
        }

    return {
        "episode_index": int(row["episode_index"]),
        "length": int(row["length"]),
        "dataset_from_index": int(row["dataset_from_index"]),
        "dataset_to_index": int(row["dataset_to_index"]),
        "left": make_segment(left_key),
        "right": make_segment(right_key),
    }


def load_episode_table(dataset_root: Path, episode_index: int, columns: list[str]) -> dict[str, Any]:
    pc, ds = import_pyarrow()
    data_root = dataset_root / "data"
    require_dir(data_root, "LeRobot data parquet")
    dataset = ds.dataset(str(data_root), format="parquet")
    schema_names = set(dataset.schema.names)
    missing = [column for column in columns if column not in schema_names]
    if missing:
        raise KeyError("LeRobot data parquet is missing columns: " + ", ".join(missing))

    table = dataset.to_table(
        columns=columns,
        filter=pc.field("episode_index") == int(episode_index),
    )
    if table.num_rows == 0:
        raise ValueError(f"No rows found for episode {episode_index}")
    if "frame_index" in table.column_names:
        table = table.sort_by([("frame_index", "ascending")])
    return table.to_pydict()


def compute_rectified_calibration(
    info: dict[str, Any],
    *,
    camera_params_key: str,
    image_key: str,
    stereo_t_scale: float,
) -> dict[str, Any]:
    params = camera_params(info, camera_params_key)
    feature = info["features"][image_key]
    height, width = int(feature["shape"][0]), int(feature["shape"][1])
    image_size = (width, height)

    left_k = np.asarray(params["camera_matrix_left"], dtype=np.float64)
    right_k = np.asarray(params["camera_matrix_right"], dtype=np.float64)
    left_dist = np.asarray(params.get("dist_coeffs_left", [0.0] * 5), dtype=np.float64)
    right_dist = np.asarray(params.get("dist_coeffs_right", [0.0] * 5), dtype=np.float64)
    right_from_left_r = np.asarray(params.get("R", np.eye(3)), dtype=np.float64)
    stereo_t = np.asarray(params["T"], dtype=np.float64) * float(stereo_t_scale)

    _, _, proj_left, proj_right, _, _, _ = cv2.stereoRectify(
        left_k,
        left_dist,
        right_k,
        right_dist,
        image_size,
        right_from_left_r,
        stereo_t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    rect_left_k = proj_left[:3, :3]
    rect_right_k = proj_right[:3, :3]
    baseline_m = abs(float(proj_right[0, 3] / proj_right[0, 0]))
    fov_degrees = float(np.degrees(2.0 * np.arctan2(width, 2.0 * rect_left_k[0, 0])))

    return {
        "image_size": [width, height],
        "baseline_m": baseline_m,
        "rectified_left_K": rect_left_k.tolist(),
        "rectified_right_K": rect_right_k.tolist(),
        "raw_left_K": left_k.tolist(),
        "raw_right_K": right_k.tolist(),
        "fov_degrees": fov_degrees,
        "stereo_t_m": stereo_t.tolist(),
        "rectified": bool(params.get("rectified", False)),
    }


def selected_frames(
    episode_length: int,
    *,
    start_frame: int,
    end_frame: int | None,
    stride: int,
    max_frames: int | None,
) -> list[int]:
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    final_end = episode_length if end_frame is None else min(end_frame, episode_length)
    if final_end <= start_frame:
        raise ValueError(f"Empty frame range: start={start_frame}, end={final_end}")
    frames = list(range(start_frame, final_end, stride))
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError("No frames selected; check start/end/stride/max-frames")
    return frames


def sanitize_scene_name(text: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text.strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("._") or "lerobot_sequence"


def default_scene_name(episode_index: int, left_key: str) -> str:
    key_suffix = left_key.replace("observation.images.", "")
    return sanitize_scene_name(f"lerobot_ep{episode_index:06d}_{key_suffix}")


def build_lerobot_manifest(
    *,
    dataset_root: Path,
    episode_index: int,
    left_key: str = DEFAULT_LEFT_KEY,
    right_key: str = DEFAULT_RIGHT_KEY,
    camera_params_key: str = DEFAULT_CAMERA_PARAMS_KEY,
    gt_column: str = DEFAULT_GT_COLUMN,
    gt_euler_order: str = "xyz",
    gt_source_frame: str = "robot_base",
    gt_pose_convention: str = "world_from_camera",
    fps: float | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    stereo_t_scale: float = 0.001,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    info = load_json(dataset_root / "meta" / "info.json")
    fps_value = float(fps or dataset_fps(info, left_key))
    episode = load_episode_info(dataset_root, episode_index, left_key, right_key, fps_value)
    frames = selected_frames(
        episode["length"],
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        max_frames=max_frames,
    )

    requested_gt_column = gt_column
    resolved_gt_column = resolve_gt_column(info, gt_column, camera_params_key)
    table = load_episode_table(
        dataset_root,
        episode["episode_index"],
        ["timestamp", "frame_index", "episode_index", resolved_gt_column],
    )
    timestamps_all = np.asarray(table["timestamp"], dtype=np.float64)
    gt_values = np.asarray(table[resolved_gt_column], dtype=np.float64)
    timestamps = timestamps_all[np.asarray(frames, dtype=np.int64)]
    gt_rows = pose6_rows_to_relative_tum(
        gt_values[np.asarray(frames, dtype=np.int64)],
        timestamps,
        gt_euler_order,
        source_frame=gt_source_frame,
        target_frame="opencv",
        pose_convention=gt_pose_convention,
    )

    calibration = compute_rectified_calibration(
        info,
        camera_params_key=camera_params_key,
        image_key=left_key,
        stereo_t_scale=stereo_t_scale,
    )
    scene_name = default_scene_name(episode["episode_index"], left_key)

    return {
        "dataset_root": str(dataset_root),
        "episode_index": int(episode["episode_index"]),
        "episode_length": int(episode["length"]),
        "fps": fps_value,
        "scene_name": scene_name,
        "left_key": left_key,
        "right_key": right_key,
        "camera_params_key": camera_params_key,
        "gt_column": resolved_gt_column,
        "gt_column_requested": requested_gt_column,
        "gt_euler_order": gt_euler_order,
        "gt_source_frame": gt_source_frame,
        "gt_pose_convention": gt_pose_convention,
        "frame_ids": [int(frame_id) for frame_id in frames],
        "timestamps": timestamps.tolist(),
        "left_video": episode["left"],
        "right_video": episode["right"],
        "left_video_frame_ids": [int(episode["left"]["start_frame"] + frame_id) for frame_id in frames],
        "right_video_frame_ids": [int(episode["right"]["start_frame"] + frame_id) for frame_id in frames],
        "gt_rows": gt_rows.tolist(),
        "calibration": calibration,
        "coordinate_note": "All trajectories use first-selected-frame-relative OpenCV optical camera poses (x right, y down, z forward).",
    }


def save_manifest(output_dir: Path, manifest: dict[str, Any]) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_json(output_dir / "manifest.json", manifest)
    gt_tum_path = save_tum(output_dir / "gt_relative_opencv_tum.txt", np.asarray(manifest["gt_rows"], dtype=np.float64))
    frame_ids_path = output_dir / "frame_ids.txt"
    np.savetxt(frame_ids_path, np.asarray(manifest["frame_ids"], dtype=np.int64), fmt="%d")
    np.savetxt(output_dir / "timestamps.txt", np.asarray(manifest["timestamps"], dtype=np.float64), fmt="%.6f")
    return manifest_path, gt_tum_path, frame_ids_path


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path.expanduser().resolve())
    manifest["frame_ids"] = [int(value) for value in manifest["frame_ids"]]
    manifest["left_video_frame_ids"] = [int(value) for value in manifest["left_video_frame_ids"]]
    manifest["right_video_frame_ids"] = [int(value) for value in manifest["right_video_frame_ids"]]
    manifest["timestamps"] = [float(value) for value in manifest["timestamps"]]
    return manifest


def iter_video_frames_pyav(
    path: Path,
    desired_frames: list[int],
    fps: float,
) -> Iterator[tuple[int, np.ndarray]]:
    import av

    if not desired_frames:
        return
    desired = [int(frame_id) for frame_id in desired_frames]
    wanted_id = 0
    first_frame = desired[0]
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if first_frame > 0:
            start_seconds = max(0.0, (first_frame - 2) / float(fps))
            container.seek(
                int(start_seconds / float(stream.time_base)),
                stream=stream,
                any_frame=False,
                backward=True,
            )
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            frame_index = int(round(float(frame.time) * float(fps)))
            while wanted_id < len(desired) and frame_index > desired[wanted_id]:
                raise RuntimeError(
                    f"Video frame {desired[wanted_id]} was skipped while decoding {path}; "
                    f"decoder advanced to {frame_index}."
                )
            if wanted_id >= len(desired):
                break
            if frame_index < desired[wanted_id]:
                continue
            image_rgb = frame.to_ndarray(format="rgb24")
            yield frame_index, np.ascontiguousarray(image_rgb)
            wanted_id += 1
            if wanted_id >= len(desired):
                break
    if wanted_id != len(desired):
        raise RuntimeError(
            f"Decoded {wanted_id}/{len(desired)} requested frames from {path}; "
            f"last requested frame was {desired[-1]}."
        )


def iter_video_frames_opencv(
    path: Path,
    desired_frames: list[int],
) -> Iterator[tuple[int, np.ndarray]]:
    if not desired_frames:
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        for frame_index in desired_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"OpenCV failed to decode frame {frame_index} from {path}")
            yield int(frame_index), cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def video_size_from_metadata(path: Path) -> tuple[int, int]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if streams:
            return int(streams[0]["width"]), int(streams[0]["height"])

    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if width > 0 and height > 0:
            return width, height
    else:
        cap.release()
    raise RuntimeError(f"Cannot determine video size for {path}")


def iter_video_frames_ffmpeg(
    path: Path,
    desired_frames: list[int],
) -> Iterator[tuple[int, np.ndarray]]:
    if not desired_frames:
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not available for AV1 decoding fallback")

    width, height = video_size_from_metadata(path)
    frame_size = width * height * 3
    first = int(desired_frames[0])
    last = int(desired_frames[-1])
    stride = int(desired_frames[1] - desired_frames[0]) if len(desired_frames) > 1 else 1
    if stride < 1 or any(int(frame) != first + idx * stride for idx, frame in enumerate(desired_frames)):
        select_terms = "+".join(f"eq(n\\,{int(frame)})" for frame in desired_frames)
    else:
        select_terms = f"between(n\\,{first}\\,{last})*not(mod(n-{first}\\,{stride}))"

    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"select={select_terms}",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if process.stdout is None:
        raise RuntimeError("Failed to open ffmpeg stdout pipe")
    emitted = 0
    try:
        for frame_index in desired_frames:
            raw = process.stdout.read(frame_size)
            if len(raw) != frame_size:
                break
            image_rgb = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            emitted += 1
            yield int(frame_index), image_rgb
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
    if emitted != len(desired_frames):
        raise RuntimeError(
            f"ffmpeg decoded {emitted}/{len(desired_frames)} requested frames from {path}"
        )


def iter_video_frames(
    path: Path,
    desired_frames: list[int],
    fps: float | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    desired = [int(frame_id) for frame_id in desired_frames]
    if not desired:
        return
    if fps is not None:
        try:
            yield from iter_video_frames_pyav(path, desired, float(fps))
            return
        except Exception as pyav_exc:
            print(f"PyAV decode failed for {path}: {pyav_exc}")
    try:
        yield from iter_video_frames_ffmpeg(path, desired)
        return
    except Exception as ffmpeg_exc:
        print(f"ffmpeg decode failed for {path}: {ffmpeg_exc}")
    yield from iter_video_frames_opencv(path, desired)


def read_video_frame(path: Path, frame_id: int, fps: float | None = None) -> np.ndarray:
    for _, image_rgb in iter_video_frames(path, [int(frame_id)], fps):
        return image_rgb
    raise RuntimeError(f"Could not decode frame {frame_id} from {path}")


def extract_frames_to_directory(
    path: Path,
    frame_ids: list[int],
    output_dir: Path,
    fps: float | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.glob("*.png"):
        old_path.unlink()
    written: list[Path] = []
    desired_frames = [int(frame_id) for frame_id in frame_ids]
    for index, (_, image_rgb) in enumerate(iter_video_frames(path, desired_frames, fps)):
        output_path = output_dir / f"{index:06d}.png"
        frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_path), frame_bgr):
            raise IOError(f"Failed to write frame: {output_path}")
        written.append(output_path)
    if len(written) != len(desired_frames):
        raise RuntimeError(
            f"Decoded {len(written)}/{len(desired_frames)} requested frames from {path}"
        )
    return written


def write_intrinsics_txt(path: Path, intrinsic: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(intrinsic, dtype=np.float64), fmt="%.9f")
    return path
