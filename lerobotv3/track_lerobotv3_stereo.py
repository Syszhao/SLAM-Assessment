#!/usr/bin/env python3
"""Run PyCuVSLAM stereo tracking on a LeRobot v3 dataset.

The default configuration targets the dataset layout under ``~/test_data`` and
uses the head stereo videos described by ``meta/info.json``:

  videos/observation.images.head_stereo_left/...
  videos/observation.images.head_stereo_right/...

Outputs are TUM trajectories, a comparison report, optional Rerun
visualization, and optional evo APE metrics when the ``evo`` package is
available in the current Python environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
from scipy.spatial.transform import Rotation as R


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path.home() / "test_data"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_LEFT_KEY = "observation.images.head_stereo_left"
DEFAULT_RIGHT_KEY = "observation.images.head_stereo_right"
DEFAULT_CAMERA_PARAMS_KEY = "head_stereo"
DEFAULT_GT_COLUMN = "auto"
TUM_FMT = "%.6f %.9f %.9f %.9f %.9f %.9f %.9f %.9f"

# LeRobot robot/base-style convention seen in many humanoid datasets:
# x forward, y left, z up. OpenCV optical camera convention:
# x right, y down, z forward.
OPENCV_FROM_ROBOT_BASE = np.eye(4, dtype=np.float64)
OPENCV_FROM_ROBOT_BASE[:3, :3] = np.asarray(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

GT_COLOR = [0, 200, 255]
CUVSLAM_COLOR = [255, 165, 0]
CUVSLAM_SE3_COLOR = [210, 90, 255]
CUVSLAM_SIM3_COLOR = [80, 220, 120]
AXIS_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]


@dataclass
class VideoSegment:
    key: str
    path: Path
    from_timestamp: float
    to_timestamp: float
    start_frame: int
    file_frame_count: int | None


@dataclass
class EpisodeInfo:
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    left: VideoSegment
    right: VideoSegment


def require_file(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, path.name)
    return json.loads(path.read_text(encoding="utf-8"))


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
            raise KeyError(f"GT column {requested!r} is not in meta/info.json features. Pose6 candidates: {candidates}")
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


def print_pose6_features(info: dict[str, Any]) -> None:
    columns = pose6_feature_columns(info)
    print("Pose6 columns in meta/info.json:")
    for column in columns:
        print(f"  {column}")
    if not columns:
        print("  <none>")


def video_path(dataset_root: Path, key: str, chunk_index: int, file_index: int) -> Path:
    return (
        dataset_root
        / "videos"
        / key
        / f"chunk-{int(chunk_index):03d}"
        / f"file-{int(file_index):03d}.mp4"
    )


def video_frame_count(path: Path) -> int | None:
    try:
        import av

        with av.open(str(path)) as container:
            frames = int(container.streams.video[0].frames or 0)
        return frames or None
    except Exception:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return frames or None


def load_episode_info(
    dataset_root: Path,
    episode_index: int,
    left_key: str,
    right_key: str,
    fps: float,
) -> EpisodeInfo:
    episode_root = dataset_root / "meta" / "episodes"
    require_file(dataset_root / "meta" / "info.json", "LeRobot metadata")
    if not episode_root.exists():
        raise FileNotFoundError(f"Missing LeRobot episode metadata directory: {episode_root}")

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
        raise KeyError(
            "Episode metadata is missing required columns for the selected video keys: "
            + ", ".join(missing)
        )

    table = dataset.to_table(
        columns=columns,
        filter=pc.field("episode_index") == int(episode_index),
    )
    if table.num_rows != 1:
        raise ValueError(f"Expected one metadata row for episode {episode_index}, got {table.num_rows}")
    row = {key: values[0] for key, values in table.to_pydict().items()}

    def make_segment(key: str) -> VideoSegment:
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
        return VideoSegment(
            key=key,
            path=path,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            start_frame=int(round(from_timestamp * fps)),
            file_frame_count=video_frame_count(path),
        )

    return EpisodeInfo(
        episode_index=int(row["episode_index"]),
        length=int(row["length"]),
        dataset_from_index=int(row["dataset_from_index"]),
        dataset_to_index=int(row["dataset_to_index"]),
        left=make_segment(left_key),
        right=make_segment(right_key),
    )


def load_episode_table(
    dataset_root: Path,
    episode_index: int,
    columns: list[str],
) -> dict[str, Any]:
    data_root = dataset_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"Missing LeRobot data directory: {data_root}")
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


def matrix_to_tum_row(timestamp: float, transform: np.ndarray) -> list[float]:
    quaternion = R.from_matrix(transform[:3, :3]).as_quat()
    quaternion /= np.linalg.norm(quaternion)
    return [float(timestamp), *transform[:3, 3].tolist(), *quaternion.tolist()]


def tum_row_to_matrix(row: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = row[1:4]
    transform[:3, :3] = R.from_quat(row[4:8]).as_matrix()
    return transform


def pose6_to_matrix(pose: np.ndarray, euler_order: str) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Expected pose vector [x y z roll pitch yaw], got shape {pose.shape}")
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


def change_pose_basis(transform: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    return target_from_source @ transform @ np.linalg.inv(target_from_source)


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


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    quaternions[valid] /= norms[valid]
    return quaternions


def is_valid_row(row: np.ndarray | None) -> bool:
    return row is not None and np.isfinite(np.asarray(row[1:8], dtype=np.float64)).all()


def load_tum(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise ValueError(f"TUM file must have 8 columns: {path}")
    rows[:, 4:8] = normalize_quaternions(rows[:, 4:8])
    return rows


def save_tum(path: Path, rows: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(rows, dtype=np.float64), fmt=TUM_FMT)
    return path


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
    reference_rows: np.ndarray | None,
    estimate_rows: np.ndarray | None,
    fps: float,
) -> dict[str, float | int] | None:
    if reference_rows is None or estimate_rows is None or len(reference_rows) == 0 or len(estimate_rows) == 0:
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


def estimate_umeyama_similarity(
    reference_rows: np.ndarray,
    estimate_rows: np.ndarray,
    fps: float,
    *,
    with_scale: bool,
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

    scale = 1.0
    if with_scale:
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


def frame_id_from_time(timestamp: float, fps: float) -> int:
    return int(round(float(timestamp) * fps))


def rows_by_frame_id(rows: np.ndarray | None, fps: float) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    if rows is None:
        return lookup
    for row in np.asarray(rows, dtype=np.float64):
        if is_valid_row(row):
            lookup[frame_id_from_time(row[0], fps)] = row
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


def rr_image(rr: Any, image_rgb: np.ndarray) -> Any:
    image = rr.Image(np.ascontiguousarray(image_rgb))
    if hasattr(image, "compress"):
        return image.compress(jpeg_quality=80)
    return image


def set_rerun_time(rr: Any, frame_id: int, timestamp: float) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", frame_id)
        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds("time", timestamp)
        return
    rr.set_time("frame", sequence=frame_id)
    rr.set_time("time", duration=timestamp)


def init_rerun(args: argparse.Namespace) -> Any | None:
    if args.no_rerun:
        return None
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ModuleNotFoundError as exc:
        print(f"Missing rerun-sdk; skipping Rerun visualization: {exc}")
        return None

    if args.save_rrd:
        Path(args.save_rrd).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    rr.init(args.rerun_app_id, spawn=args.rerun_spawn and not args.save_rrd)
    if args.save_rrd:
        rr.save(str(Path(args.save_rrd).expanduser().resolve()))
    try:
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    except Exception as exc:  # pragma: no cover - depends on rerun version
        print(f"Rerun coordinate setup failed, using defaults: {exc}")

    try:
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Horizontal(
                    column_shares=[0.42, 0.58],
                    contents=[
                        rrb.Vertical(
                            contents=[
                                rrb.Spatial2DView(name="Left", origin="world/input/left"),
                                rrb.Spatial2DView(name="Right", origin="world/input/right"),
                            ]
                        ),
                        rrb.Spatial3DView(name="GT vs cuVSLAM", origin="world"),
                    ],
                ),
            ),
            make_active=True,
        )
    except Exception as exc:  # pragma: no cover - depends on rerun version
        print(f"Rerun blueprint setup failed, using default layout: {exc}")
    return rr


def log_static_trajectory(
    rr: Any,
    entity_path: str,
    rows: np.ndarray | None,
    color: list[int],
    radius: float,
) -> None:
    if rr is None:
        return
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
) -> None:
    if rr is None or row is None or not is_valid_row(row):
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


def log_observations(rr: Any, observations: list[Any], image_path: str) -> None:
    if rr is None or not observations:
        return
    points = np.asarray([[obs.u, obs.v] for obs in observations], dtype=np.float32)
    colors = np.asarray(
        [[(obs.id * 17) % 256, (obs.id * 31) % 256, (obs.id * 47) % 256] for obs in observations],
        dtype=np.uint8,
    )
    rr.log(f"{image_path}/observations", rr.Points2D(points, colors=colors, radii=3.0))


def iter_video_frames_pyav(
    path: Path,
    desired_frames: list[int],
    fps: float,
) -> Iterator[tuple[int, np.ndarray]]:
    import av

    if not desired_frames:
        return
    desired = list(desired_frames)
    wanted_id = 0
    first_frame = desired[0]
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if first_frame > 0:
            start_seconds = max(0.0, (first_frame - 2) / fps)
            container.seek(
                int(start_seconds / float(stream.time_base)),
                stream=stream,
                any_frame=False,
                backward=True,
            )
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            frame_index = int(round(float(frame.time) * fps))
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
            yield frame_index, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def video_size_from_metadata(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if width > 0 and height > 0:
            return width, height
    else:
        cap.release()

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError(f"Cannot determine video size for {path}; ffprobe is not available")
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
    if not streams:
        raise RuntimeError(f"ffprobe did not find a video stream in {path}")
    return int(streams[0]["width"]), int(streams[0]["height"])


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
    first = desired_frames[0]
    last = desired_frames[-1]
    stride = desired_frames[1] - desired_frames[0] if len(desired_frames) > 1 else 1
    if stride < 1 or any(frame != first + idx * stride for idx, frame in enumerate(desired_frames)):
        select_terms = "+".join(f"eq(n\\,{frame})" for frame in desired_frames)
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
            yield frame_index, image_rgb
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
    fps: float,
) -> Iterator[tuple[int, np.ndarray]]:
    try:
        yield from iter_video_frames_pyav(path, desired_frames, fps)
    except Exception as pyav_exc:
        try:
            yield from iter_video_frames_ffmpeg(path, desired_frames)
        except Exception as ffmpeg_exc:
            print(f"PyAV decode failed for {path}: {pyav_exc}")
            print(f"ffmpeg decode failed for {path}: {ffmpeg_exc}")
            yield from iter_video_frames_opencv(path, desired_frames)


def rgb_to_tracker_image(image_rgb: np.ndarray, image_format: str) -> np.ndarray:
    if image_format == "gray":
        return np.ascontiguousarray(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY))
    if image_format == "bgr":
        return np.ascontiguousarray(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return np.ascontiguousarray(image_rgb)


def brown_distortion_from_opencv(coeffs: list[float]) -> list[float]:
    coeffs = [float(value) for value in coeffs]
    if len(coeffs) >= 5:
        k1, k2, p1, p2, k3 = coeffs[:5]
        return [k1, k2, k3, p1, p2]
    return coeffs


def pose_from_rotation_translation(vslam: Any, rotation: np.ndarray, translation: np.ndarray) -> Any:
    quaternion = R.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat()
    quaternion /= np.linalg.norm(quaternion)
    return vslam.Pose(
        rotation=quaternion.tolist(),
        translation=np.asarray(translation, dtype=np.float64).tolist(),
    )


def create_cuvslam_stereo_tracker(
    info: dict[str, Any],
    args: argparse.Namespace,
) -> Any:
    try:
        import cuvslam
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python module `cuvslam`. Run this script in the PyCuVSLAM "
            "environment, or install the wheel built from cuVSLAM/python."
        ) from exc

    params = camera_params(info, args.camera_params_key)

    left_k = np.asarray(params["camera_matrix_left"], dtype=np.float64)
    right_k = np.asarray(params["camera_matrix_right"], dtype=np.float64)
    left_dist = params.get("dist_coeffs_left", [])
    right_dist = params.get("dist_coeffs_right", [])

    feature = info["features"][args.left_key]
    height, width = int(feature["shape"][0]), int(feature["shape"][1])
    image_size = (width, height)

    def make_camera(k_matrix: np.ndarray, dist_coeffs: list[float]) -> Any:
        camera = cuvslam.Camera()
        camera.size = image_size
        camera.focal = [float(k_matrix[0, 0]), float(k_matrix[1, 1])]
        camera.principal = [float(k_matrix[0, 2]), float(k_matrix[1, 2])]
        if args.rectified_stereo:
            camera.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole)
        else:
            camera.distortion = cuvslam.Distortion(
                cuvslam.Distortion.Model.Brown,
                brown_distortion_from_opencv(dist_coeffs),
            )
        camera.border_top = args.border_top
        camera.border_bottom = args.border_bottom
        camera.border_left = args.border_left
        camera.border_right = args.border_right
        return camera

    left_camera = make_camera(left_k, left_dist)
    right_camera = make_camera(right_k, right_dist)

    if args.baseline_m is not None:
        right_translation = np.asarray([float(args.baseline_m), 0.0, 0.0], dtype=np.float64)
        right_rotation = np.eye(3, dtype=np.float64)
    else:
        stereo_t = np.asarray(params["T"], dtype=np.float64) * float(args.stereo_t_scale)
        if args.rectified_stereo and not args.use_full_stereo_extrinsics:
            baseline = abs(float(stereo_t[0])) if abs(float(stereo_t[0])) > 1e-9 else float(np.linalg.norm(stereo_t))
            right_translation = np.asarray([baseline, 0.0, 0.0], dtype=np.float64)
            right_rotation = np.eye(3, dtype=np.float64)
        else:
            right_from_left_r = np.asarray(params.get("R", np.eye(3)), dtype=np.float64)
            right_rotation = right_from_left_r.T
            right_translation = -right_from_left_r.T @ stereo_t

    right_camera.rig_from_camera = pose_from_rotation_translation(
        cuvslam,
        right_rotation,
        right_translation,
    )

    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=args.async_sba,
        enable_observations_export=args.log_observations,
        enable_final_landmarks_export=False,
        rectified_stereo_camera=args.rectified_stereo,
        odometry_mode=cuvslam.Tracker.OdometryMode.Multicamera,
    )
    cfg.use_gpu = not args.no_gpu
    cfg.use_denoising = args.use_denoising
    cfg.max_frame_delta_s = args.max_frame_delta_s

    rig = cuvslam.Rig([left_camera, right_camera])
    print(
        "cuVSLAM stereo rig: "
        f"size={image_size}, right_translation={right_translation.tolist()}, "
        f"rectified={args.rectified_stereo}, image_format={args.image_format}"
    )
    return cuvslam.Tracker(rig, cfg)


def run_cuvslam_stereo(
    info: dict[str, Any],
    episode: EpisodeInfo,
    selected_episode_frames: list[int],
    timestamps: np.ndarray,
    gt_rows: np.ndarray | None,
    args: argparse.Namespace,
    rr: Any | None,
) -> tuple[np.ndarray, list[int]]:
    tracker = create_cuvslam_stereo_tracker(info, args)
    left_desired = [episode.left.start_frame + frame_id for frame_id in selected_episode_frames]
    right_desired = [episode.right.start_frame + frame_id for frame_id in selected_episode_frames]

    left_iter = iter_video_frames(episode.left.path, left_desired, args.fps)
    right_iter = iter_video_frames(episode.right.path, right_desired, args.fps)
    gt_lookup = rows_by_frame_id(gt_rows, args.fps) if gt_rows is not None else {}

    rows: list[list[float]] = []
    failed_frames: list[int] = []
    trajectory: list[np.ndarray] = []

    for pair_index, ((_, left_rgb), (_, right_rgb)) in enumerate(zip(left_iter, right_iter)):
        episode_frame = selected_episode_frames[pair_index]
        timestamp_s = float(timestamps[pair_index])
        timestamp_ns = int(round(timestamp_s * 1e9))
        left_image = rgb_to_tracker_image(left_rgb, args.image_format)
        right_image = rgb_to_tracker_image(right_rgb, args.image_format)

        pose_estimate, _ = tracker.track(timestamp_ns, (left_image, right_image))
        row = None
        if pose_estimate.world_from_rig is None:
            failed_frames.append(episode_frame)
        else:
            pose = pose_estimate.world_from_rig.pose
            row = [
                timestamp_s,
                *np.asarray(pose.translation, dtype=np.float64).tolist(),
                *normalize_quaternions(np.asarray(pose.rotation, dtype=np.float64)[None, :])[0].tolist(),
            ]
            rows.append(row)
            trajectory.append(np.asarray(row[1:4], dtype=np.float64))

        if rr is not None and (pair_index % args.rerun_stride == 0):
            set_rerun_time(rr, episode_frame, timestamp_s)
            rr.log("world/input/left", rr_image(rr, left_rgb))
            rr.log("world/input/right", rr_image(rr, right_rgb))
            log_pose(
                rr,
                "world/gt/current",
                gt_lookup.get(frame_id_from_time(timestamp_s, args.fps)),
                GT_COLOR,
                args.rerun_axis_length,
            )
            log_pose(rr, "world/cuvslam_raw/current", np.asarray(row) if row else None, CUVSLAM_COLOR, args.rerun_axis_length)
            if len(trajectory) > 1:
                rr.log(
                    "world/cuvslam_raw/live_trajectory",
                    rr.LineStrips3D(np.asarray(trajectory), colors=[CUVSLAM_COLOR], radii=args.rerun_trajectory_radius),
                )
            if args.log_observations and row is not None:
                try:
                    log_observations(rr, tracker.get_last_observations(0), "world/input/left")
                except Exception as exc:
                    print(f"Could not log cuVSLAM observations: {exc}")

        if (pair_index + 1) % max(args.progress_every, 1) == 0:
            print(f"Tracked {pair_index + 1}/{len(selected_episode_frames)} stereo frames")

    if len(rows) + len(failed_frames) != len(selected_episode_frames):
        decoded = len(rows) + len(failed_frames)
        print(f"Warning: decoded/tracked {decoded}/{len(selected_episode_frames)} selected frame pairs.")

    print(f"cuVSLAM tracking complete: valid={len(rows)}, failed={len(failed_frames)}")
    return np.asarray(rows, dtype=np.float64), failed_frames


def maybe_run_evo(
    gt_tum_path: Path | None,
    estimate_paths: dict[str, Path],
    output_dir: Path,
    fps: float,
    skip_evo: bool,
) -> Path | None:
    if skip_evo or gt_tum_path is None:
        return None
    try:
        from evo.core import metrics, sync
        from evo.tools import file_interface
    except ModuleNotFoundError as exc:
        print(f"Missing evo dependency; skipping evo metrics: {exc.name or 'evo'}")
        return None

    evo_dir = output_dir / "evo_summary"
    evo_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"results": {}}
    traj_ref_full = file_interface.read_tum_trajectory_file(str(gt_tum_path))

    for variant, estimate_path in estimate_paths.items():
        traj_est_full = file_interface.read_tum_trajectory_file(str(estimate_path))
        traj_ref, traj_est = sync.associate_trajectories(
            copy.deepcopy(traj_ref_full),
            traj_est_full,
            max_diff=0.5 / fps + 1e-6,
        )
        variant_results: dict[str, Any] = {"matched_poses": traj_ref.num_poses}
        for name, relation in (
            ("position_m", metrics.PoseRelation.translation_part),
            ("orientation_deg", metrics.PoseRelation.rotation_angle_deg),
        ):
            ape = metrics.APE(relation)
            ape.process_data((traj_ref, traj_est))
            result = ape.get_result()
            variant_results[name] = {
                "title": result.info.get("title", name),
                "stats": {key: float(value) for key, value in result.stats.items()},
            }
            np.savetxt(
                evo_dir / f"{variant}_ape_{name}_errors.txt",
                result.np_arrays["error_array"],
                fmt="%.9f",
            )
        results["results"][variant] = variant_results

    metrics_path = evo_dir / "evo_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved evo metrics: {metrics_path}")
    return metrics_path


def write_report(
    output_dir: Path,
    *,
    dataset_root: Path,
    episode: EpisodeInfo,
    selected_episode_frames: list[int],
    args: argparse.Namespace,
    paths: dict[str, str],
    failed_frames: list[int],
    raw_stats: dict[str, float | int] | None,
    se3_stats: dict[str, float | int] | None,
    sim3_stats: dict[str, float | int] | None,
    se3_similarity: dict[str, Any] | None,
    sim3_similarity: dict[str, Any] | None,
) -> Path:
    report = {
        "dataset_root": str(dataset_root),
        "episode_index": episode.episode_index,
        "episode_length": episode.length,
        "selected_frames": len(selected_episode_frames),
        "first_selected_frame": selected_episode_frames[0],
        "last_selected_frame": selected_episode_frames[-1],
        "fps": args.fps,
        "left_video": str(episode.left.path),
        "right_video": str(episode.right.path),
        "left_video_start_frame": episode.left.start_frame,
        "right_video_start_frame": episode.right.start_frame,
        "left_key": args.left_key,
        "right_key": args.right_key,
        "gt_column": None if args.no_gt else args.gt_column,
        "gt_column_requested": None if args.no_gt else getattr(args, "gt_column_requested", args.gt_column),
        "gt_euler_order": args.gt_euler_order,
        "gt_source_frame": args.gt_source_frame,
        "gt_pose_convention": args.gt_pose_convention,
        "trajectory_frame": "opencv",
        "coordinate_note": (
            "All TUM rows are first-selected-frame-relative OpenCV optical camera poses "
            "(x right, y down, z forward). cuVSLAM reports the selected rig frame in this basis. "
            "When --gt-column=auto, the GT pose column is selected from meta/info.json pose6 "
            "features, preferring the D435/head camera state for head_stereo data."
        ),
        "cuvslam": {
            "camera_params_key": args.camera_params_key,
            "rectified_stereo": args.rectified_stereo,
            "stereo_t_scale": args.stereo_t_scale,
            "baseline_m_override": args.baseline_m,
            "use_full_stereo_extrinsics": args.use_full_stereo_extrinsics,
            "image_format": args.image_format,
        },
        "failed_frames": failed_frames,
        "raw_stats": raw_stats,
        "se3_stats": se3_stats,
        "sim3_stats": sim3_stats,
        "se3_similarity": se3_similarity,
        "sim3_similarity": sim3_similarity,
        "paths": paths,
    }
    report_json = output_dir / "comparison_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report_txt = output_dir / "comparison_report.txt"
    with report_txt.open("w", encoding="utf-8") as file:
        file.write("LeRobot v3 stereo cuVSLAM report\n")
        file.write(f"dataset_root={dataset_root}\n")
        file.write(f"episode_index={episode.episode_index}\n")
        file.write(
            f"frames={len(selected_episode_frames)} "
            f"{selected_episode_frames[0]}->{selected_episode_frames[-1]} fps={args.fps:.6f}\n"
        )
        file.write(f"left_video={episode.left.path}\n")
        file.write(f"right_video={episode.right.path}\n")
        file.write(f"gt_column={None if args.no_gt else args.gt_column}\n")
        if not args.no_gt:
            file.write(f"gt_column_requested={getattr(args, 'gt_column_requested', args.gt_column)}\n")
        file.write(f"gt_source_frame={args.gt_source_frame}\n")
        file.write(f"gt_pose_convention={args.gt_pose_convention}\n")
        file.write("trajectory_frame=opencv (x right, y down, z forward)\n")
        file.write(f"camera_params_key={args.camera_params_key}\n")
        file.write(f"rectified_stereo={args.rectified_stereo}\n")
        file.write(f"failed_frames={len(failed_frames)}\n\n")

        for name, stats in (("raw", raw_stats), ("se3", se3_stats), ("sim3", sim3_stats)):
            file.write(f"[{name}]\n")
            if stats is None:
                file.write("stats=None\n\n")
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
    return report_txt


def similarity_report(similarity: dict[str, np.ndarray | float] | None) -> dict[str, Any] | None:
    if similarity is None:
        return None
    rotation = np.asarray(similarity["rotation"], dtype=np.float64)
    return {
        "scale": float(similarity["scale"]),
        "rotation_angle_deg": float(np.degrees(R.from_matrix(rotation).magnitude())),
        "translation": np.asarray(similarity["translation"], dtype=np.float64).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cuVSLAM stereo odometry on a LeRobot v3 episode, visualize in Rerun, and evaluate with evo."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", type=Path, help="output directory; default is outputs/episode_<id>")
    parser.add_argument("--left-key", default=DEFAULT_LEFT_KEY)
    parser.add_argument("--right-key", default=DEFAULT_RIGHT_KEY)
    parser.add_argument("--camera-params-key", default=DEFAULT_CAMERA_PARAMS_KEY)
    parser.add_argument("--fps", type=float, help="override dataset/video FPS")
    parser.add_argument("--start-frame", type=int, default=0, help="episode-relative start frame")
    parser.add_argument("--end-frame", type=int, help="episode-relative exclusive end frame")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, help="debug limit after start/end/stride selection")
    parser.add_argument("--prepare-only", action="store_true", help="write frame metadata/GT and stop")
    parser.add_argument(
        "--skip-tracking",
        action="store_true",
        help="reuse existing cuvslam_stereo_opencv_tum.txt, falling back to legacy cuvslam_stereo_raw_tum.txt",
    )

    parser.add_argument(
        "--gt-column",
        default=DEFAULT_GT_COLUMN,
        help=(
            "Pose6 GT column, or 'auto' to select the camera pose column from "
            "meta/info.json. For the default G1 head stereo data this resolves "
            "to observation.state.state_d435."
        ),
    )
    parser.add_argument(
        "--list-pose-columns",
        action="store_true",
        help="print [x y z roll pitch yaw] pose columns discovered in meta/info.json and exit",
    )
    parser.add_argument("--gt-euler-order", default="xyz")
    parser.add_argument(
        "--gt-source-frame",
        choices=("robot_base", "opencv"),
        default="robot_base",
        help=(
            "coordinate basis used by --gt-column. robot_base means x forward, y left, z up; "
            "it is converted to OpenCV optical x right, y down, z forward before evaluation."
        ),
    )
    parser.add_argument(
        "--gt-pose-convention",
        choices=("world_from_camera", "camera_from_world"),
        default="world_from_camera",
        help="absolute transform convention represented by --gt-column before relative conversion",
    )
    parser.add_argument("--no-gt", action="store_true", help="do not load GT/evo comparison")
    parser.add_argument("--skip-evo", action="store_true")
    parser.add_argument(
        "--evo-include-sim3-diagnostic",
        action="store_true",
        help="also run evo on Sim3-scaled diagnostic output; leave off for metric stereo evaluation",
    )

    parser.add_argument(
        "--rectified-stereo",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "whether cuVSLAM should use rectified stereo mode; default follows "
            "camera_params.<key>.rectified from meta/info.json"
        ),
    )
    parser.add_argument(
        "--stereo-t-scale",
        type=float,
        default=0.001,
        help="scale for camera_params head_stereo T; default converts millimeters to meters",
    )
    parser.add_argument("--baseline-m", type=float, help="override stereo baseline in meters")
    parser.add_argument(
        "--use-full-stereo-extrinsics",
        action="store_true",
        help="use R,T from calibration instead of identity + horizontal baseline for rectified stereo",
    )
    parser.add_argument("--image-format", choices=("gray", "bgr", "rgb"), default="gray")
    parser.add_argument("--async-sba", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--use-denoising", action="store_true")
    parser.add_argument("--max-frame-delta-s", type=float, default=1.0)
    parser.add_argument("--border-top", type=int, default=8)
    parser.add_argument("--border-bottom", type=int, default=8)
    parser.add_argument("--border-left", type=int, default=8)
    parser.add_argument("--border-right", type=int, default=8)
    parser.add_argument("--log-observations", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)

    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument("--rerun-app-id", default="lerobotv3_cuvslam_stereo")
    parser.add_argument("--rerun-spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rrd", help="save a .rrd recording instead of spawning the viewer")
    parser.add_argument("--rerun-axis-length", type=float, default=0.04)
    parser.add_argument("--rerun-trajectory-radius", type=float, default=0.002)
    parser.add_argument("--rerun-stride", type=int, default=1, help="log every Nth processed frame to Rerun")
    return parser.parse_args()


def selected_frames(args: argparse.Namespace, episode_length: int) -> list[int]:
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    end_frame = episode_length if args.end_frame is None else min(args.end_frame, episode_length)
    if end_frame <= args.start_frame:
        raise ValueError(f"Empty frame range: start={args.start_frame}, end={end_frame}")
    frames = list(range(args.start_frame, end_frame, args.stride))
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise ValueError("No frames selected; check --start-frame/--end-frame/--stride/--max-frames")
    return frames


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    info = load_json(dataset_root / "meta" / "info.json")
    if args.list_pose_columns:
        print_pose6_features(info)
        return

    params = camera_params(info, args.camera_params_key)
    requested_gt_column = args.gt_column
    if not args.no_gt:
        args.gt_column = resolve_gt_column(info, args.gt_column, args.camera_params_key)
    args.gt_column_requested = requested_gt_column
    if args.rectified_stereo is None:
        args.rectified_stereo = bool(params.get("rectified", False))

    args.fps = float(args.fps or dataset_fps(info, args.left_key))

    episode = load_episode_info(
        dataset_root,
        args.episode_index,
        args.left_key,
        args.right_key,
        args.fps,
    )
    frames = selected_frames(args, episode.length)

    output_dir = (
        args.output.expanduser().resolve()
        if args.output
        else (DEFAULT_OUTPUT_ROOT / f"episode_{args.episode_index:06d}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rrd is None and not args.no_rerun:
        args.save_rrd = str(output_dir / "cuvslam_stereo.rrd")

    columns = ["timestamp", "frame_index", "episode_index"]
    if not args.no_gt:
        columns.append(args.gt_column)
    data = load_episode_table(dataset_root, episode.episode_index, columns)
    frame_index = np.asarray(data["frame_index"], dtype=np.int64)
    timestamps_all = np.asarray(data["timestamp"], dtype=np.float64)
    if len(frame_index) != episode.length:
        print(f"Warning: metadata length={episode.length}, data rows={len(frame_index)}")
    max_selected = max(frames)
    if max_selected >= len(timestamps_all):
        raise ValueError(f"Selected frame {max_selected} exceeds episode table length {len(timestamps_all)}")

    timestamps = timestamps_all[np.asarray(frames, dtype=np.int64)]
    np.savetxt(output_dir / "frame_ids.txt", np.asarray(frames, dtype=np.int64), fmt="%d")
    np.savetxt(output_dir / "timestamps.txt", timestamps, fmt="%.9f")

    gt_rows = None
    gt_tum_path = None
    if not args.no_gt:
        gt_values = np.asarray(data[args.gt_column], dtype=np.float64)
        gt_rows = pose6_rows_to_relative_tum(
            gt_values[np.asarray(frames, dtype=np.int64)],
            timestamps,
            args.gt_euler_order,
            source_frame=args.gt_source_frame,
            target_frame="opencv",
            pose_convention=args.gt_pose_convention,
        )
        gt_tum_path = save_tum(output_dir / "gt_relative_opencv_tum.txt", gt_rows)
        save_tum(output_dir / "gt_relative_tum.txt", gt_rows)
        print(f"Saved GT OpenCV TUM: {gt_tum_path}")

    print(f"Dataset: {dataset_root}")
    print(f"Episode {episode.episode_index}: {episode.length} frames, selected {len(frames)}")
    print(f"Left video: {episode.left.path} @ frame {episode.left.start_frame}")
    print(f"Right video: {episode.right.path} @ frame {episode.right.start_frame}")
    if not args.no_gt:
        print(f"GT pose column: {args.gt_column} (requested {requested_gt_column})")
    print(f"camera_params: {args.camera_params_key}, rectified_stereo={args.rectified_stereo}")
    print(f"Output directory: {output_dir}")

    if args.prepare_only:
        print("prepare-only complete; cuVSLAM/Rerun/evo were not run.")
        return

    rr = init_rerun(args)
    log_static_trajectory(rr, "world/gt", gt_rows, GT_COLOR, args.rerun_trajectory_radius)

    raw_tum_path = output_dir / "cuvslam_stereo_opencv_tum.txt"
    failed_frames: list[int] = []
    if args.skip_tracking:
        if not raw_tum_path.exists():
            raw_tum_path = output_dir / "cuvslam_stereo_raw_tum.txt"
        require_file(raw_tum_path, "existing cuVSLAM OpenCV TUM for --skip-tracking")
        cuvslam_rows = load_tum(raw_tum_path)
        print(f"Reusing cuVSLAM OpenCV TUM: {raw_tum_path}")
    else:
        cuvslam_rows, failed_frames = run_cuvslam_stereo(
            info,
            episode,
            frames,
            timestamps,
            gt_rows,
            args,
            rr,
        )
        if len(cuvslam_rows) == 0:
            raise RuntimeError("cuVSLAM did not return any valid poses")
        raw_tum_path = save_tum(raw_tum_path, cuvslam_rows)
        save_tum(output_dir / "cuvslam_stereo_raw_tum.txt", cuvslam_rows)
        print(f"Saved cuVSLAM OpenCV TUM: {raw_tum_path}")

    se3_rows = None
    sim3_rows = None
    se3_tum_path = None
    sim3_tum_path = None
    se3_similarity = None
    sim3_similarity = None
    if gt_rows is not None:
        se3_similarity = estimate_umeyama_similarity(gt_rows, cuvslam_rows, args.fps, with_scale=False)
        sim3_similarity = estimate_umeyama_similarity(gt_rows, cuvslam_rows, args.fps, with_scale=True)
        if se3_similarity is not None:
            se3_rows = apply_similarity_to_tum_rows(cuvslam_rows, se3_similarity)
            se3_tum_path = save_tum(output_dir / "cuvslam_stereo_opencv_se3_aligned_tum.txt", se3_rows)
            save_tum(output_dir / "cuvslam_stereo_se3_aligned_tum.txt", se3_rows)
        if sim3_similarity is not None:
            sim3_rows = apply_similarity_to_tum_rows(cuvslam_rows, sim3_similarity)
            sim3_tum_path = save_tum(output_dir / "cuvslam_stereo_opencv_sim3_aligned_tum.txt", sim3_rows)
            save_tum(output_dir / "cuvslam_stereo_sim3_aligned_tum.txt", sim3_rows)

    log_static_trajectory(rr, "world/cuvslam_raw", cuvslam_rows, CUVSLAM_COLOR, args.rerun_trajectory_radius)
    log_static_trajectory(rr, "world/cuvslam_se3", se3_rows, CUVSLAM_SE3_COLOR, args.rerun_trajectory_radius)
    log_static_trajectory(rr, "world/cuvslam_sim3", sim3_rows, CUVSLAM_SIM3_COLOR, args.rerun_trajectory_radius)

    raw_stats = trajectory_stats(gt_rows, cuvslam_rows, args.fps)
    se3_stats = trajectory_stats(gt_rows, se3_rows, args.fps)
    sim3_stats = trajectory_stats(gt_rows, sim3_rows, args.fps)

    paths: dict[str, str] = {
        "frame_ids": str(output_dir / "frame_ids.txt"),
        "timestamps": str(output_dir / "timestamps.txt"),
        "cuvslam_opencv_tum": str(raw_tum_path),
    }
    if gt_tum_path is not None:
        paths["gt_opencv_tum"] = str(gt_tum_path)
    if se3_tum_path is not None:
        paths["cuvslam_opencv_se3_tum"] = str(se3_tum_path)
    if sim3_tum_path is not None:
        paths["cuvslam_opencv_sim3_tum"] = str(sim3_tum_path)
    if args.save_rrd:
        paths["rerun_rrd"] = str(Path(args.save_rrd).expanduser().resolve())

    evo_inputs = {"raw": raw_tum_path}
    if se3_tum_path is not None:
        evo_inputs["se3"] = se3_tum_path
    if args.evo_include_sim3_diagnostic and sim3_tum_path is not None:
        evo_inputs["sim3_diagnostic"] = sim3_tum_path
    evo_path = maybe_run_evo(gt_tum_path, evo_inputs, output_dir, args.fps, args.skip_evo)
    if evo_path is not None:
        paths["evo_metrics"] = str(evo_path)

    report_path = write_report(
        output_dir,
        dataset_root=dataset_root,
        episode=episode,
        selected_episode_frames=frames,
        args=args,
        paths=paths,
        failed_frames=failed_frames,
        raw_stats=raw_stats,
        se3_stats=se3_stats,
        sim3_stats=sim3_stats,
        se3_similarity=similarity_report(se3_similarity),
        sim3_similarity=similarity_report(sim3_similarity),
    )
    print(f"Saved comparison report: {report_path}")

    for label, stats in (("Raw", raw_stats), ("SE3", se3_stats), ("Sim3", sim3_stats)):
        if stats is None:
            continue
        print(
            f"{label} cuVSLAM | "
            f"translation RMSE {stats['translation_rmse']:.6f} m | "
            f"rotation RMSE {stats['rotation_rmse']:.3f} deg"
        )
    if args.save_rrd:
        print(f"Saved Rerun recording: {args.save_rrd}")


if __name__ == "__main__":
    main()
