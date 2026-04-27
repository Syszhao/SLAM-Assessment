#!/usr/bin/env python3
"""Run EgoDex cuVSLAM, DROID-SLAM, and MegaSAM, then compare them in Rerun.

Given an EgoDex hdf5/mp4 pair, this script invokes the three per-method demo
pipelines, reads their exported trajectories, converts everything into one
first-frame-relative OpenCV camera basis, logs the comparison to Rerun, and
saves evo Sim(3)-aligned pose plots.

* MegaSAM: first-frame-relative OpenCV TUM.
* DROID-SLAM: first-frame-relative OpenCV TUM, usually scale-only aligned.
* cuVSLAM: ARKit world TUM, optionally scale-only aligned.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HDF5 = Path("/home/user/test/add_remove_lid/0.hdf5")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "egodex_all_slam_rerun_outputs"

OPENCV_FROM_ARKIT_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])
TUM_FMT = "%.6f %.9f %.9f %.9f %.9f %.9f %.9f %.9f"

GT_COLOR = [0, 200, 255]
CUSLAM_COLOR = [255, 170, 0]
DROID_COLOR = [210, 90, 255]
MEGASAM_COLOR = [255, 90, 90]
AXIS_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]


@dataclass
class Trajectory:
    key: str
    label: str
    rows: np.ndarray
    color: list[int]
    source_path: Path
    note: str
    orientation_valid: bool = True


def sanitize_scene_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "egodex_sequence"


def default_scene_name(hdf5_path: Path) -> str:
    return sanitize_scene_name(f"{hdf5_path.parent.name}_{hdf5_path.stem}")


def resolve_path(path: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    base_candidate = (base / path).resolve()
    cwd_candidate = path.resolve()
    if base_candidate.exists() or not cwd_candidate.exists():
        return base_candidate
    return cwd_candidate


def paired_mp4_path(hdf5_path: Path) -> Path:
    return hdf5_path.with_suffix(".mp4")


def read_video_info(mp4_path: Path | None) -> dict[str, float | int] | None:
    if mp4_path is None or not mp4_path.exists():
        return None
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
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
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = pose[:3]
        transform[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
        return transform
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


def load_egodex_poses(hdf5_path: Path) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as root:
        pose_data = load_camera_poses(root)
    return np.stack([pose_data_to_matrix(pose) for pose in pose_data])


def load_egodex_intrinsic(hdf5_path: Path) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as root:
        intrinsic = root["camera/intrinsic"][:].astype(np.float64)
    if intrinsic.shape != (3, 3):
        intrinsic = intrinsic.reshape(3, 3)
    return intrinsic


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
    transform[:3, 3] = row[1:4]
    transform[:3, :3] = R.from_quat(row[4:8]).as_matrix()
    return transform


def change_pose_basis(transform: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    return target_from_source @ transform @ np.linalg.inv(target_from_source)


def is_valid_row(row: np.ndarray | None) -> bool:
    return row is not None and np.isfinite(np.asarray(row[1:8], dtype=np.float64)).all()


def load_tum(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise ValueError(f"TUM file must have 8 columns: {path}")
    rows = rows[np.isfinite(rows[:, 1:8]).all(axis=1)]
    if len(rows) == 0:
        raise ValueError(f"No valid rows in TUM file: {path}")
    rows[:, 4:8] = normalize_quaternions(rows[:, 4:8])
    return rows


def save_tum(path: Path, rows: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(rows, dtype=np.float64), fmt=TUM_FMT)
    return path


def frame_id_from_time(timestamp: float, fps: float) -> int:
    return int(round(float(timestamp) * fps))


def frame_ids_from_rows(rows: np.ndarray, fps: float) -> list[int]:
    return [frame_id_from_time(row[0], fps) for row in np.asarray(rows, dtype=np.float64)]


def rows_by_frame_id(rows: np.ndarray | None, fps: float) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    if rows is None:
        return lookup
    for row in np.asarray(rows, dtype=np.float64):
        if is_valid_row(row):
            lookup[frame_id_from_time(row[0], fps)] = row
    return lookup


def copy_orientations_by_frame_id(
    position_rows: np.ndarray,
    orientation_rows: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, int]:
    orientation_lookup = rows_by_frame_id(orientation_rows, fps)
    fixed_rows = np.asarray(position_rows, dtype=np.float64).copy()
    copied = 0
    for index, row in enumerate(fixed_rows):
        orientation_row = orientation_lookup.get(frame_id_from_time(row[0], fps))
        if orientation_row is None or not is_valid_row(orientation_row):
            continue
        fixed_rows[index, 4:8] = orientation_row[4:8]
        copied += 1
    fixed_rows[:, 4:8] = normalize_quaternions(fixed_rows[:, 4:8])
    return fixed_rows, copied


def relative_gt_rows(
    world_from_camera_arkit: np.ndarray,
    fps: float,
    origin_frame_id: int,
    frame_ids: list[int],
) -> np.ndarray:
    camera_origin_from_world = np.linalg.inv(world_from_camera_arkit[origin_frame_id])
    rows = []
    for frame_id in frame_ids:
        camera_origin_from_camera_arkit = camera_origin_from_world @ world_from_camera_arkit[frame_id]
        camera_origin_from_camera_opencv = change_pose_basis(
            camera_origin_from_camera_arkit,
            OPENCV_FROM_ARKIT_CAMERA,
        )
        rows.append(matrix_to_tum_row(frame_id / fps, camera_origin_from_camera_opencv))
    return np.asarray(rows, dtype=np.float64)


def arkit_world_rows_to_relative_opencv(
    rows: np.ndarray,
    world_from_camera_origin_arkit: np.ndarray,
) -> np.ndarray:
    camera_origin_from_world = np.linalg.inv(world_from_camera_origin_arkit)
    converted = []
    for row in np.asarray(rows, dtype=np.float64):
        world_from_camera_arkit = tum_row_to_matrix(row)
        camera_origin_from_camera_arkit = camera_origin_from_world @ world_from_camera_arkit
        camera_origin_from_camera_opencv = change_pose_basis(
            camera_origin_from_camera_arkit,
            OPENCV_FROM_ARKIT_CAMERA,
        )
        converted.append(matrix_to_tum_row(row[0], camera_origin_from_camera_opencv))
    return np.asarray(converted, dtype=np.float64)


def rebase_relative_rows(
    rows: np.ndarray,
    fps: float,
    origin_frame_id: int,
) -> tuple[np.ndarray, str | None]:
    lookup = rows_by_frame_id(rows, fps)
    origin_row = lookup.get(origin_frame_id)
    if origin_row is None:
        return (
            np.asarray(rows, dtype=np.float64).copy(),
            f"origin frame {origin_frame_id} not present; kept native relative origin",
        )

    old_origin_from_new_origin = tum_row_to_matrix(origin_row)
    new_origin_from_old_origin = np.linalg.inv(old_origin_from_new_origin)
    rebased = []
    for row in np.asarray(rows, dtype=np.float64):
        new_pose = new_origin_from_old_origin @ tum_row_to_matrix(row)
        rebased.append(matrix_to_tum_row(row[0], new_pose))
    return np.asarray(rebased, dtype=np.float64), None


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def command_prefix(command: str) -> list[str]:
    return shlex.split(command)


def conda_env_python_candidates(env_names: list[str]) -> list[Path]:
    env_roots: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        if prefix.parent.name == "envs":
            env_roots.append(prefix.parent)
        else:
            env_roots.append(prefix / "envs")

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        env_roots.append(Path(conda_exe).resolve().parents[1] / "envs")

    executable = Path(sys.executable).resolve()
    if len(executable.parents) >= 3 and executable.parents[1].parent.name == "envs":
        env_roots.append(executable.parents[1].parent)
    if len(executable.parents) >= 2:
        env_roots.append(executable.parents[1] / "envs")

    env_roots.append(Path.home() / "miniconda3" / "envs")
    env_roots.append(Path.home() / "anaconda3" / "envs")

    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in env_roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        for name in env_names:
            candidate = root / name / "bin" / "python"
            if candidate.exists():
                candidates.append(candidate)
    return candidates


def python_command_works(command: str, import_check: str, cwd: Path) -> bool:
    try:
        result = subprocess.run(
            [*command_prefix(command), "-c", import_check],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return False
    return result.returncode == 0


def resolve_python_command(
    requested: str,
    *,
    label: str,
    env_var: str,
    env_names: list[str],
    import_check: str,
    cwd: Path,
) -> str:
    override = os.environ.get(env_var)
    if requested != "auto":
        return requested
    if override:
        if python_command_works(override, import_check, cwd):
            print(f"Auto-selected {label} Python from ${env_var}: {override}")
            return override
        print(f"Ignoring ${env_var}; import check failed: {override}")

    current = sys.executable
    current_env_name = Path(current).resolve().parents[1].name if len(Path(current).resolve().parents) > 1 else ""
    if current_env_name in env_names and python_command_works(current, import_check, cwd):
        print(f"Auto-selected {label} Python from current environment: {current}")
        return current

    for candidate in conda_env_python_candidates(env_names):
        command = str(candidate)
        if python_command_works(command, import_check, cwd):
            print(f"Auto-selected {label} Python: {command}")
            return command

    if python_command_works(current, import_check, cwd):
        print(f"Auto-selected {label} Python from current environment: {current}")
        return current

    names = ", ".join(env_names)
    raise RuntimeError(
        f"Could not auto-select a Python for {label}. Tried current env and conda env names: {names}. "
        f"Pass --{label}-python explicitly, e.g. `--{label}-python \"conda run -n {env_names[0]} python\"`."
    )


def run_command(
    command: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    continue_on_error: bool,
) -> bool:
    print("$ " + " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return True
    try:
        subprocess.run(command, cwd=str(cwd), check=True)
        return True
    except subprocess.CalledProcessError as exc:
        if not continue_on_error:
            raise
        print(f"Command failed with exit code {exc.returncode}; continuing because --continue-on-error is set.")
        return False


def append_common_frame_args(
    command: list[str],
    *,
    fps: float,
    start_frame: int,
    end_frame: int | None,
    max_frames: int | None,
    stride: int,
) -> None:
    command.extend(
        [
            "--fps",
            str(fps),
            "--start-frame",
            str(start_frame),
            "--stride",
            str(stride),
        ]
    )
    if end_frame is not None:
        command.extend(["--end-frame", str(end_frame)])
    if max_frames is not None:
        command.extend(["--max-frames", str(max_frames)])


def expected_output_ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_slam_pipelines(
    args: argparse.Namespace,
    *,
    hdf5_path: Path,
    mp4_path: Path,
    scene_name: str,
    output_dir: Path,
    fps: float,
) -> dict[str, Path]:
    run_root = output_dir / "runs"
    output_dirs = {
        "cuslam": run_root / "cuslam",
        "droid": run_root / "droid",
        "megasam": run_root / "megasam",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    if not args.skip_cuslam:
        cuslam_python = resolve_python_command(
            args.cuslam_python,
            label="cuslam",
            env_var="EGODEX_CUSLAM_PYTHON",
            env_names=["cuvslam"],
            import_check="import cuvslam",
            cwd=REPO_ROOT,
        )
        expected = output_dirs["cuslam"] / "cuslam_arkit_world_scale_only_tum.txt"
        if args.reuse_existing and expected_output_ready(expected):
            print(f"Reusing cuVSLAM output: {expected}")
        else:
            command = [
                *command_prefix(cuslam_python),
                str(REPO_ROOT / "cuVSLAM" / "examples" / "ego-dex" / "egodex_vo.py"),
                "--hdf5",
                str(hdf5_path),
                "--mp4",
                str(mp4_path),
                "--output",
                str(output_dirs["cuslam"]),
                "--no-rerun",
            ]
            append_common_frame_args(
                command,
                fps=fps,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                max_frames=args.max_frames,
                stride=args.stride,
            )
            command.extend(shlex.split(args.cuslam_extra_args))
            run_command(
                command,
                cwd=REPO_ROOT,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            )

    if not args.skip_droid:
        droid_python = resolve_python_command(
            args.droid_python,
            label="droid",
            env_var="EGODEX_DROID_PYTHON",
            env_names=["droid-slam", "droid_slam"],
            import_check="import torch; import droid_backends",
            cwd=REPO_ROOT / "DROID-SLAM",
        )
        expected = output_dirs["droid"] / "droid_relative_opencv_scale_only_tum.txt"
        if args.reuse_existing and expected_output_ready(expected):
            print(f"Reusing DROID-SLAM output: {expected}")
        else:
            command = [
                *command_prefix(droid_python),
                str(REPO_ROOT / "DROID-SLAM" / "egodex_rerun.py"),
                "--hdf5",
                str(hdf5_path),
                "--mp4",
                str(mp4_path),
                "--output",
                str(output_dirs["droid"]),
                "--no-rerun",
                "--skip-evo",
            ]
            append_common_frame_args(
                command,
                fps=fps,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                max_frames=args.max_frames,
                stride=args.stride,
            )
            command.extend(shlex.split(args.droid_extra_args))
            run_command(
                command,
                cwd=REPO_ROOT,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            )

    if not args.skip_megasam:
        megasam_python = resolve_python_command(
            args.megasam_python,
            label="megasam",
            env_var="EGODEX_MEGASAM_PYTHON",
            env_names=["mega_sam", "megasam", "mega-sam"],
            import_check="import h5py; import cv2; import torch",
            cwd=REPO_ROOT / "mega-sam",
        )
        expected = output_dirs["megasam"] / "megasam_relative_opencv_tum.txt"
        if args.reuse_existing and expected_output_ready(expected):
            print(f"Reusing MegaSAM output: {expected}")
        else:
            command = [
                *command_prefix(megasam_python),
                str(REPO_ROOT / "mega-sam" / "egodex_megasam_rerun.py"),
                "--hdf5",
                str(hdf5_path),
                "--mp4",
                str(mp4_path),
                "--output",
                str(output_dirs["megasam"]),
                "--scene-name",
                scene_name,
                "--no-rerun",
                "--skip-evo",
                "--python",
                megasam_python,
            ]
            append_common_frame_args(
                command,
                fps=fps,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                max_frames=args.max_frames,
                stride=args.stride,
            )
            command.extend(shlex.split(args.megasam_extra_args))
            run_command(
                command,
                cwd=REPO_ROOT,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            )

    return output_dirs


def select_megasam_path(args: argparse.Namespace, scene_name: str) -> tuple[Path | None, str]:
    if args.megasam_tum:
        return resolve_path(args.megasam_tum), "explicit MegaSAM TUM"

    megasam_dir = resolve_path(
        args.megasam_dir or REPO_ROOT / "mega-sam" / "egodex_outputs" / scene_name
    )
    if args.megasam_variant == "raw":
        names = ["megasam_relative_opencv_tum.txt"]
    elif args.megasam_variant == "sim3":
        names = ["megasam_relative_opencv_sim3_tum.txt", "megasam_relative_opencv_tum.txt"]
    else:
        names = ["megasam_relative_opencv_sim3_tum.txt", "megasam_relative_opencv_tum.txt"]
    return first_existing([megasam_dir / name for name in names]), f"MegaSAM dir {megasam_dir}"


def select_droid_path(args: argparse.Namespace, scene_name: str) -> tuple[Path | None, str]:
    if args.droid_tum:
        return resolve_path(args.droid_tum), "explicit DROID-SLAM TUM"

    droid_dir = resolve_path(
        args.droid_dir or REPO_ROOT / "DROID-SLAM" / "egodex_outputs" / scene_name
    )
    if args.prefer_scale:
        names = ["droid_relative_opencv_scale_only_tum.txt", "droid_relative_opencv_tum.txt"]
    else:
        names = ["droid_relative_opencv_tum.txt", "droid_relative_opencv_scale_only_tum.txt"]
    return first_existing([droid_dir / name for name in names]), f"DROID-SLAM dir {droid_dir}"


def select_cuslam_path(args: argparse.Namespace) -> tuple[Path | None, str]:
    if args.cuslam_tum:
        return resolve_path(args.cuslam_tum), "explicit cuVSLAM TUM"

    cuslam_dir = resolve_path(args.cuslam_dir or REPO_ROOT / "cuVSLAM" / "examples" / "ego-dex" / "egodex_tum")
    if args.prefer_scale:
        names = ["cuslam_arkit_world_scale_only_tum.txt", "cuslam_arkit_world_tum.txt"]
    else:
        names = ["cuslam_arkit_world_tum.txt", "cuslam_arkit_world_scale_only_tum.txt"]
    return first_existing([cuslam_dir / name for name in names]), f"cuVSLAM dir {cuslam_dir}"


def choose_origin_frame(
    trajectories: list[np.ndarray],
    fps: float,
    requested_origin: int | None,
) -> tuple[int, list[str]]:
    if requested_origin is not None:
        return requested_origin, []

    frame_sets = [set(frame_ids_from_rows(rows, fps)) for rows in trajectories if rows is not None]
    if not frame_sets:
        return 0, []

    common = set.intersection(*frame_sets)
    if 0 in common:
        return 0, []
    if common:
        return min(common), []

    first_frames = [min(frame_set) for frame_set in frame_sets if frame_set]
    origin = min(first_frames) if first_frames else 0
    return origin, [
        "No common frame exists across all loaded trajectories; "
        f"using frame {origin} where possible."
    ]


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
    estimate_rows: np.ndarray,
    fps: float,
    *,
    include_rotation: bool = True,
) -> dict[str, float | int] | None:
    ref, est = match_by_timestamp(reference_rows, estimate_rows, max_time_diff=0.5 / fps + 1e-6)
    if len(ref) == 0:
        return None
    finite = np.isfinite(ref[:, 1:8]).all(axis=1) & np.isfinite(est[:, 1:8]).all(axis=1)
    ref = ref[finite]
    est = est[finite]
    if len(ref) == 0:
        return None
    translation_errors = np.linalg.norm(est[:, 1:4] - ref[:, 1:4], axis=1)
    stats: dict[str, float | int] = {
        "frames": int(len(ref)),
        "translation_rmse": float(np.sqrt(np.mean(translation_errors**2))),
        "translation_mean": float(np.mean(translation_errors)),
        "translation_max": float(np.max(translation_errors)),
    }
    if include_rotation:
        rotation_errors = rotation_error_deg(ref, est)
        stats.update(
            {
                "rotation_rmse": float(np.sqrt(np.mean(rotation_errors**2))),
                "rotation_mean": float(np.mean(rotation_errors)),
                "rotation_max": float(np.max(rotation_errors)),
            }
        )
    return stats


def evo_quat_wxyz_to_xyzw(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    return quaternions[:, [1, 2, 3, 0]]


def evo_trajectory_series(traj: Any, *, time_zero: float) -> dict[str, np.ndarray]:
    positions = np.asarray(traj.positions_xyz, dtype=np.float64)
    quaternions = normalize_quaternions(evo_quat_wxyz_to_xyzw(traj.orientations_quat_wxyz))
    euler_deg = np.degrees(np.unwrap(R.from_quat(quaternions).as_euler("xyz", degrees=False), axis=0))
    timestamps = getattr(traj, "timestamps", None)
    if timestamps is None:
        time = np.arange(len(positions), dtype=np.float64)
    else:
        time = np.asarray(timestamps, dtype=np.float64) - time_zero
    return {"time": time, "position": positions, "euler_deg": euler_deg}


def evo_traj_to_tum_rows(traj: Any) -> np.ndarray:
    positions = np.asarray(traj.positions_xyz, dtype=np.float64)
    quaternions_xyzw = normalize_quaternions(evo_quat_wxyz_to_xyzw(traj.orientations_quat_wxyz))
    timestamps = getattr(traj, "timestamps", None)
    if timestamps is None:
        timestamps = np.arange(len(positions), dtype=np.float64)
    else:
        timestamps = np.asarray(timestamps, dtype=np.float64)
    return np.column_stack((timestamps, positions, quaternions_xyzw))


def tum_rows_series(rows: np.ndarray, *, time_zero: float) -> dict[str, np.ndarray]:
    rows = np.asarray(rows, dtype=np.float64)
    quaternions = normalize_quaternions(rows[:, 4:8])
    euler_deg = np.degrees(np.unwrap(R.from_quat(quaternions).as_euler("xyz", degrees=False), axis=0))
    return {
        "time": rows[:, 0] - time_zero,
        "position": rows[:, 1:4],
        "euler_deg": euler_deg,
    }


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


def apply_similarity_to_rows(
    rows: np.ndarray,
    similarity: dict[str, np.ndarray | float],
    *,
    rotate_orientation: bool = True,
) -> np.ndarray:
    scale = float(similarity["scale"])
    rotation = np.asarray(similarity["rotation"], dtype=np.float64)
    translation = np.asarray(similarity["translation"], dtype=np.float64)
    aligned = []
    for row in np.asarray(rows, dtype=np.float64):
        transform = tum_row_to_matrix(row)
        transform[:3, 3] = scale * (rotation @ transform[:3, 3]) + translation
        if rotate_orientation:
            transform[:3, :3] = rotation @ transform[:3, :3]
        aligned.append(matrix_to_tum_row(row[0], transform))
    return np.asarray(aligned, dtype=np.float64)


def plot_aligned_pose_series(
    output_dir: Path,
    gt_series: dict[str, np.ndarray],
    aligned_series: dict[str, dict[str, Any]],
    *,
    title: str,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis_names = ("x", "y", "z")
    angle_names = ("roll", "pitch", "yaw")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    fig.suptitle(title, y=1.02)
    for idx, axis_name in enumerate(axis_names):
        ax = axes[0, idx]
        ax.plot(gt_series["time"], gt_series["position"][:, idx], color="black", label="GT", linewidth=2.2)
        for item in aligned_series.values():
            series = item["series"]
            ax.plot(series["time"], series["position"][:, idx], label=item["label"], color=item["color"], linewidth=1.7)
        ax.set_title(f"position {axis_name}")
        ax.set_ylabel("m")
        ax.grid(True, alpha=0.3)

    for idx, angle_name in enumerate(angle_names):
        ax = axes[1, idx]
        ax.plot(gt_series["time"], gt_series["euler_deg"][:, idx], color="black", label="GT", linewidth=2.2)
        for item in aligned_series.values():
            if not item.get("orientation_valid", True):
                continue
            series = item["series"]
            ax.plot(series["time"], series["euler_deg"][:, idx], label=item["label"], color=item["color"], linewidth=1.7)
        ax.set_title(angle_name)
        ax.set_ylabel("deg")
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    pose_path = output_dir / "evo_aligned_pose_comparison.png"
    fig.savefig(pose_path, dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        gt_series["position"][:, 0],
        gt_series["position"][:, 1],
        gt_series["position"][:, 2],
        color="black",
        label="GT",
        linewidth=2.2,
    )
    for item in aligned_series.values():
        series = item["series"]
        ax.plot(
            series["position"][:, 0],
            series["position"][:, 1],
            series["position"][:, 2],
            label=item["label"],
            color=item["color"],
            linewidth=1.7,
        )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="best")
    ax.set_title("Sim(3)-aligned trajectories")
    fig.tight_layout()
    trajectory_path = output_dir / "evo_aligned_trajectory_xyz.png"
    fig.savefig(trajectory_path, dpi=180)
    plt.close(fig)

    return {"pose_comparison": str(pose_path), "trajectory_xyz": str(trajectory_path)}


def save_umeyama_aligned_pose_plots(
    gt_tum_path: Path,
    trajectories: list[Trajectory],
    converted_paths: dict[str, str],
    output_dir: Path,
    fps: float,
    *,
    reason: str,
) -> dict[str, Any]:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError as exc:
        print(f"Missing plotting dependency; skipping Sim(3)-aligned pose plots: {exc.name}")
        return {}

    evo_dir = output_dir / "evo_aligned"
    evo_dir.mkdir(parents=True, exist_ok=True)
    gt_rows = load_tum(gt_tum_path)
    time_zero = float(gt_rows[0, 0])
    gt_series = tum_rows_series(gt_rows, time_zero=time_zero)

    aligned_series: dict[str, dict[str, Any]] = {}
    results: dict[str, Any] = {
        "reference_tum": str(gt_tum_path),
        "backend": "umeyama_fallback",
        "fallback_reason": reason,
        "correct_scale": True,
        "trajectories": {},
    }
    for trajectory in trajectories:
        converted_path = converted_paths.get(trajectory.key)
        if converted_path is None:
            continue
        rows = load_tum(Path(converted_path))
        similarity = estimate_umeyama_similarity(gt_rows, rows, fps)
        if similarity is None:
            results["trajectories"][trajectory.key] = {
                "source_tum": converted_path,
                "error": "not enough matched poses for Sim(3) alignment",
            }
            continue
        aligned_rows = apply_similarity_to_rows(
            rows,
            similarity,
            rotate_orientation=trajectory.key != "megasam",
        )
        aligned_path = save_tum(evo_dir / f"{trajectory.key}_umeyama_sim3_tum.txt", aligned_rows)
        aligned_series[trajectory.key] = {
            "label": trajectory.label,
            "color": np.asarray(trajectory.color, dtype=np.float64) / 255.0,
            "series": tum_rows_series(aligned_rows, time_zero=time_zero),
            "orientation_valid": trajectory.orientation_valid,
        }
        stats = trajectory_stats(gt_rows, aligned_rows, fps, include_rotation=trajectory.orientation_valid)
        results["trajectories"][trajectory.key] = {
            "source_tum": converted_path,
            "aligned_tum": str(aligned_path),
            "alignment": {
                "type": "Sim(3)",
                "scale": float(similarity["scale"]),
                "rotation": np.asarray(similarity["rotation"], dtype=np.float64).tolist(),
                "translation": np.asarray(similarity["translation"], dtype=np.float64).tolist(),
            },
            "stats_vs_gt_after_alignment": stats,
        }

    if aligned_series:
        results["plots"] = plot_aligned_pose_series(
            evo_dir,
            gt_series,
            aligned_series,
            title="EgoDex pose comparison after Sim(3) alignment (evo fallback)",
        )
        print(f"Saved Sim(3)-aligned pose plot: {results['plots']['pose_comparison']}")

    metrics_path = evo_dir / "evo_aligned_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return {"metrics": str(metrics_path), "plots": results.get("plots", {}), "results": results}


def save_evo_aligned_pose_plots(
    gt_tum_path: Path,
    trajectories: list[Trajectory],
    converted_paths: dict[str, str],
    output_dir: Path,
    fps: float,
    *,
    skip: bool,
) -> dict[str, Any]:
    if skip or not trajectories:
        return {}

    try:
        from evo.core import metrics, sync
        from evo.tools import file_interface
    except ModuleNotFoundError as exc:
        reason = f"missing evo dependency: {exc.name or 'evo'}"
        print(f"Missing evo dependency; using Umeyama Sim(3) fallback for aligned pose plots: {exc.name or 'evo'}")
        return save_umeyama_aligned_pose_plots(
            gt_tum_path,
            trajectories,
            converted_paths,
            output_dir,
            fps,
            reason=reason,
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"Missing plotting dependency; skipping evo-aligned pose plots: {exc.name}")
        return {}

    evo_dir = output_dir / "evo_aligned"
    evo_dir.mkdir(parents=True, exist_ok=True)

    traj_ref_full = file_interface.read_tum_trajectory_file(str(gt_tum_path))
    gt_rows_for_stats = load_tum(gt_tum_path)
    ref_timestamps = getattr(traj_ref_full, "timestamps", None)
    time_zero = float(np.asarray(ref_timestamps, dtype=np.float64)[0]) if ref_timestamps is not None else 0.0
    gt_series = evo_trajectory_series(traj_ref_full, time_zero=time_zero)

    aligned_series: dict[str, dict[str, Any]] = {}
    results: dict[str, Any] = {
        "reference_tum": str(gt_tum_path),
        "backend": "evo",
        "correct_scale": True,
        "trajectories": {},
    }

    for trajectory in trajectories:
        converted_path = converted_paths.get(trajectory.key)
        if converted_path is None:
            continue

        try:
            traj_ref = file_interface.read_tum_trajectory_file(str(gt_tum_path))
            traj_est = file_interface.read_tum_trajectory_file(converted_path)
            traj_ref, traj_est = sync.associate_trajectories(
                traj_ref,
                traj_est,
                max_diff=0.5 / fps + 1e-6,
            )
            traj_est_eval = copy.deepcopy(traj_est)
            rotation, translation, scale = traj_est_eval.align(traj_ref, correct_scale=True)
            aligned_rows = evo_traj_to_tum_rows(traj_est_eval)
            if trajectory.key == "megasam":
                source_rows = load_tum(Path(converted_path))
                aligned_rows, _ = copy_orientations_by_frame_id(aligned_rows, source_rows, fps)
            aligned_path = save_tum(evo_dir / f"{trajectory.key}_evo_sim3_tum.txt", aligned_rows)
            aligned_series[trajectory.key] = {
                "label": trajectory.label,
                "color": np.asarray(trajectory.color, dtype=np.float64) / 255.0,
                "series": tum_rows_series(aligned_rows, time_zero=time_zero),
                "orientation_valid": trajectory.orientation_valid,
            }

            trajectory_result: dict[str, Any] = {
                "source_tum": converted_path,
                "aligned_tum": str(aligned_path),
                "matched_poses": traj_ref.num_poses,
                "alignment": {
                    "type": "Sim(3)",
                    "scale": float(scale),
                    "rotation": np.asarray(rotation, dtype=np.float64).tolist(),
                    "translation": np.asarray(translation, dtype=np.float64).tolist(),
                },
                "results": {},
                "row_stats_vs_gt_after_alignment": trajectory_stats(
                    gt_rows_for_stats,
                    aligned_rows,
                    fps,
                    include_rotation=trajectory.orientation_valid,
                ),
            }
            for name, relation in (
                ("position_m", metrics.PoseRelation.translation_part),
                ("orientation_deg", metrics.PoseRelation.rotation_angle_deg),
            ):
                if name == "orientation_deg" and trajectory.key == "megasam":
                    trajectory_result["results"][name] = {
                        "title": "rotation_angle_deg_from_preserved_megasam_quaternion",
                        "stats": trajectory_result["row_stats_vs_gt_after_alignment"],
                    }
                    continue
                ape = metrics.APE(relation)
                ape.process_data((traj_ref, traj_est_eval))
                result = ape.get_result()
                trajectory_result["results"][name] = {
                    "title": result.info.get("title", name),
                    "stats": {key: float(value) for key, value in result.stats.items()},
                }
                np.savetxt(
                    evo_dir / f"{trajectory.key}_ape_{name}_errors.txt",
                    np.asarray(result.np_arrays["error_array"], dtype=np.float64),
                    fmt="%.9f",
                )
            results["trajectories"][trajectory.key] = trajectory_result
        except Exception as exc:  # pragma: no cover - depends on evo/local data
            results["trajectories"][trajectory.key] = {
                "source_tum": converted_path,
                "error": str(exc),
            }
            print(f"Failed evo alignment for {trajectory.label}: {exc}")

    if not aligned_series:
        metrics_path = evo_dir / "evo_aligned_metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return {"metrics": str(metrics_path), "results": results}

    axis_names = ("x", "y", "z")
    angle_names = ("roll", "pitch", "yaw")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    fig.suptitle("EgoDex pose comparison after evo Sim(3) alignment", y=1.02)
    for idx, axis_name in enumerate(axis_names):
        ax = axes[0, idx]
        ax.plot(gt_series["time"], gt_series["position"][:, idx], color="black", label="GT", linewidth=2.2)
        for item in aligned_series.values():
            series = item["series"]
            ax.plot(series["time"], series["position"][:, idx], label=item["label"], color=item["color"], linewidth=1.7)
        ax.set_title(f"position {axis_name}")
        ax.set_ylabel("m")
        ax.grid(True, alpha=0.3)

    for idx, angle_name in enumerate(angle_names):
        ax = axes[1, idx]
        ax.plot(gt_series["time"], gt_series["euler_deg"][:, idx], color="black", label="GT", linewidth=2.2)
        for item in aligned_series.values():
            if not item.get("orientation_valid", True):
                continue
            series = item["series"]
            ax.plot(series["time"], series["euler_deg"][:, idx], label=item["label"], color=item["color"], linewidth=1.7)
        ax.set_title(angle_name)
        ax.set_ylabel("deg")
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    pose_path = evo_dir / "evo_aligned_pose_comparison.png"
    fig.savefig(pose_path, dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        gt_series["position"][:, 0],
        gt_series["position"][:, 1],
        gt_series["position"][:, 2],
        color="black",
        label="GT",
        linewidth=2.2,
    )
    for item in aligned_series.values():
        series = item["series"]
        ax.plot(
            series["position"][:, 0],
            series["position"][:, 1],
            series["position"][:, 2],
            label=item["label"],
            color=item["color"],
            linewidth=1.7,
        )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="best")
    ax.set_title("evo Sim(3)-aligned trajectories")
    fig.tight_layout()
    trajectory_path = evo_dir / "evo_aligned_trajectory_xyz.png"
    fig.savefig(trajectory_path, dpi=180)
    plt.close(fig)

    results["plots"] = {
        "pose_comparison": str(pose_path),
        "trajectory_xyz": str(trajectory_path),
    }
    metrics_path = evo_dir / "evo_aligned_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved evo-aligned pose plot: {pose_path}")
    return {"metrics": str(metrics_path), "plots": results["plots"], "results": results}


def build_rerun_trajectories(
    trajectories: list[Trajectory],
    evo_aligned: dict[str, Any],
    mode: str,
) -> list[Trajectory]:
    if mode == "selected" or not evo_aligned:
        return trajectories

    aligned: list[Trajectory] = []
    aligned_info = evo_aligned.get("results", {}).get("trajectories", {})
    for trajectory in trajectories:
        info = aligned_info.get(trajectory.key, {})
        aligned_tum = info.get("aligned_tum")
        if not aligned_tum:
            continue
        aligned_path = Path(aligned_tum)
        if not aligned_path.exists():
            continue
        aligned.append(
            Trajectory(
                key=f"{trajectory.key}_sim3",
                label=f"{trajectory.label} Sim(3)",
                rows=load_tum(aligned_path),
                color=trajectory.color,
                source_path=aligned_path,
                note=f"Rerun display trajectory after Sim(3) alignment from {trajectory.source_path}",
                orientation_valid=trajectory.orientation_valid,
            )
        )

    if mode == "sim3":
        return aligned or trajectories

    if mode == "both":
        originals = [
            Trajectory(
                key=f"{trajectory.key}_selected",
                label=f"{trajectory.label} selected",
                rows=trajectory.rows,
                color=trajectory.color,
                source_path=trajectory.source_path,
                note=trajectory.note,
                orientation_valid=trajectory.orientation_valid,
            )
            for trajectory in trajectories
        ]
        return [*originals, *aligned]

    return trajectories


def trajectory_points(rows: np.ndarray | None) -> np.ndarray | None:
    if rows is None:
        return None
    rows = np.asarray(rows, dtype=np.float64)
    valid = np.isfinite(rows[:, 1:8]).all(axis=1)
    points = rows[valid, 1:4]
    if len(points) < 2:
        return None
    return points


def set_rerun_time(rr: Any, frame_id: int, fps: float) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", frame_id)
        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds("time", frame_id / fps)
        return
    rr.set_time("frame", sequence=frame_id)
    rr.set_time("time", duration=frame_id / fps)


def rr_image(rr: Any, image_rgb: np.ndarray) -> Any:
    image = rr.Image(image_rgb)
    if hasattr(image, "compress"):
        return image.compress(jpeg_quality=80)
    return image


def log_static_trajectory(
    rr: Any,
    entity_path: str,
    rows: np.ndarray,
    color: list[int],
    radius: float,
) -> None:
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
    *,
    show_axes: bool = True,
) -> None:
    if row is None or not is_valid_row(row):
        return
    translation = np.asarray(row[1:4], dtype=np.float64)
    if not show_axes:
        rr.log(
            entity_path,
            rr.Points3D([translation], colors=[color], radii=origin_radius),
        )
        return
    quaternion = normalize_quaternions(np.asarray(row[4:8], dtype=np.float64)[None, :])[0]
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=translation,
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


def init_rerun(args: argparse.Namespace, output_dir: Path) -> Any:
    try:
        import rerun as rr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing rerun-sdk. Install it in the environment running this script, "
            "or pass --no-rerun to only write converted TUM/report files."
        ) from exc

    save_rrd = Path(args.save_rrd) if args.save_rrd else None
    if save_rrd is None and args.save_rrd_default:
        save_rrd = output_dir / "all_slam_comparison.rrd"
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
                        rrb.Spatial2DView(name="Video", origin="world/input/image"),
                        rrb.Spatial3DView(name="GT / cuVSLAM / DROID / MegaSAM", origin="world"),
                    ],
                ),
            ),
            make_active=True,
        )
    except Exception as exc:  # pragma: no cover - depends on rerun version
        print(f"Rerun blueprint setup failed, using default layout: {exc}")

    return rr


def read_video_frame(cap: cv2.VideoCapture, frame_id: int) -> np.ndarray | None:
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_id:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame_bgr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def visualize_in_rerun(
    gt_rows: np.ndarray,
    trajectories: list[Trajectory],
    mp4_path: Path | None,
    fps: float,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    if args.no_rerun:
        return

    rr = init_rerun(args, output_dir)
    log_static_trajectory(rr, "world/gt", gt_rows, GT_COLOR, args.rerun_trajectory_radius)
    for trajectory in trajectories:
        log_static_trajectory(
            rr,
            f"world/{trajectory.key}",
            trajectory.rows,
            trajectory.color,
            args.rerun_trajectory_radius,
        )

    gt_lookup = rows_by_frame_id(gt_rows, fps)
    lookups = {trajectory.key: rows_by_frame_id(trajectory.rows, fps) for trajectory in trajectories}
    selected_frames = set(gt_lookup)
    for lookup in lookups.values():
        selected_frames.update(lookup)
    if args.rerun_frame_stride > 1:
        selected_frames = {
            frame_id
            for frame_id in selected_frames
            if (frame_id - min(selected_frames)) % args.rerun_frame_stride == 0
        }
    if args.max_rerun_frames is not None:
        selected_frames = set(sorted(selected_frames)[: args.max_rerun_frames])

    cap = None
    if mp4_path is not None and mp4_path.exists():
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            print(f"Cannot open video; Rerun will contain trajectories only: {mp4_path}")
            cap = None

    logged_frames = 0
    for frame_id in sorted(selected_frames):
        set_rerun_time(rr, frame_id, fps)
        if cap is not None:
            image_rgb = read_video_frame(cap, frame_id)
            if image_rgb is not None:
                rr.log("world/input/image", rr_image(rr, image_rgb))

        log_pose(
            rr,
            "world/gt/current",
            gt_lookup.get(frame_id),
            GT_COLOR,
            args.rerun_axis_length,
            args.rerun_origin_radius,
        )
        for trajectory in trajectories:
            log_pose(
                rr,
                f"world/{trajectory.key}/current",
                lookups[trajectory.key].get(frame_id),
                trajectory.color,
                args.rerun_axis_length,
                args.rerun_origin_radius,
                show_axes=trajectory.orientation_valid,
            )
        logged_frames += 1

    if cap is not None:
        cap.release()
    print(f"Sent Rerun visualization: {logged_frames} frames")
    if args.save_rrd:
        print(f"Saved Rerun recording: {args.save_rrd}")
    elif args.save_rrd_default:
        print(f"Saved Rerun recording: {output_dir / 'all_slam_comparison.rrd'}")


def write_report(
    output_dir: Path,
    *,
    scene_name: str,
    hdf5_path: Path,
    mp4_path: Path | None,
    fps: float,
    origin_frame_id: int,
    gt_path: Path,
    trajectories: list[Trajectory],
    stats: dict[str, dict[str, float | int] | None],
    missing: list[str],
    warnings: list[str],
    converted_paths: dict[str, str],
    evo_aligned: dict[str, Any],
    rerun_trajectory_mode: str,
) -> Path:
    report = {
        "scene_name": scene_name,
        "hdf5": str(hdf5_path),
        "mp4": str(mp4_path) if mp4_path is not None else None,
        "fps": fps,
        "origin_frame_id": origin_frame_id,
        "coordinate": "first-frame-relative OpenCV camera basis (x right, y down, z forward)",
        "gt_tum": str(gt_path),
        "trajectories": {
            trajectory.key: {
                "label": trajectory.label,
                "source": str(trajectory.source_path),
                "note": trajectory.note,
                "converted_tum": converted_paths.get(trajectory.key),
                "stats_vs_gt": stats.get(trajectory.key),
            }
            for trajectory in trajectories
        },
        "missing": missing,
        "warnings": warnings,
        "evo_aligned": evo_aligned,
        "rerun_trajectory_mode": rerun_trajectory_mode,
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    text_path = output_dir / "comparison_report.txt"
    with text_path.open("w", encoding="utf-8") as file:
        file.write("EgoDex GT / cuVSLAM / DROID-SLAM / MegaSAM\n")
        file.write("Coordinate: first-frame-relative OpenCV camera basis (x right, y down, z forward).\n")
        file.write(f"scene_name={scene_name}\n")
        file.write(f"hdf5={hdf5_path}\n")
        if mp4_path is not None:
            file.write(f"mp4={mp4_path}\n")
        file.write(f"fps={fps:.6f}\n")
        file.write(f"origin_frame_id={origin_frame_id}\n")
        file.write(f"gt_tum={gt_path}\n\n")

        for trajectory in trajectories:
            file.write(f"[{trajectory.label}]\n")
            file.write(f"source={trajectory.source_path}\n")
            file.write(f"converted_tum={converted_paths.get(trajectory.key)}\n")
            file.write(f"note={trajectory.note}\n")
            values = stats.get(trajectory.key)
            if values is None:
                file.write("stats=None\n\n")
            else:
                line = (
                    f"frames={values['frames']} "
                    f"trans_rmse={values['translation_rmse']:.6f}m "
                    f"trans_mean={values['translation_mean']:.6f}m "
                    f"trans_max={values['translation_max']:.6f}m "
                )
                if "rotation_rmse" in values:
                    line += (
                        f"rot_rmse={values['rotation_rmse']:.3f}deg "
                        f"rot_mean={values['rotation_mean']:.3f}deg "
                        f"rot_max={values['rotation_max']:.3f}deg"
                    )
                else:
                    line += "rotation_note=not_meaningful_after_similarity_alignment"
                file.write(line + "\n\n")

        if missing:
            file.write("[missing]\n")
            for item in missing:
                file.write(f"{item}\n")
            file.write("\n")
        if warnings:
            file.write("[warnings]\n")
            for item in warnings:
                file.write(f"{item}\n")
            file.write("\n")
        if evo_aligned:
            file.write("[evo_aligned]\n")
            plots = evo_aligned.get("plots", {})
            if plots:
                for key, value in plots.items():
                    file.write(f"{key}={value}\n")
            if evo_aligned.get("metrics"):
                file.write(f"metrics={evo_aligned['metrics']}\n")
            file.write(f"rerun_trajectory_mode={rerun_trajectory_mode}\n")
    return text_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EgoDex cuVSLAM, DROID-SLAM, and MegaSAM, then compare them with GT in one Rerun scene."
    )
    parser.add_argument("--hdf5", default=str(DEFAULT_HDF5), help="EgoDex .hdf5 path")
    parser.add_argument("--mp4", help="EgoDex .mp4 path; default is the hdf5 path with .mp4 suffix")
    parser.add_argument("--fps", type=float, help="override video FPS; default reads from mp4 or uses 30")
    parser.add_argument("--scene-name", help="default: <hdf5-parent>_<hdf5-stem>")
    parser.add_argument("--output", help="output dir for converted TUM/report files")
    parser.add_argument("--origin-frame", type=int, help="common trajectory origin frame; default auto")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="exclusive end frame")
    parser.add_argument("--max-frames", type=int, help="debug limit after stride/start-frame filtering")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")

    parser.add_argument(
        "--run-slam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run cuVSLAM, DROID-SLAM, and MegaSAM before the final comparison",
    )
    parser.add_argument("--reuse-existing", action="store_true", help="reuse per-method outputs when present")
    parser.add_argument("--dry-run", action="store_true", help="print SLAM commands without running them")
    parser.add_argument("--continue-on-error", action="store_true", help="continue final aggregation if one SLAM command fails")
    parser.add_argument("--skip-cuslam", action="store_true", help="do not run/read cuVSLAM")
    parser.add_argument("--skip-droid", action="store_true", help="do not run/read DROID-SLAM")
    parser.add_argument("--skip-megasam", action="store_true", help="do not run/read MegaSAM")
    parser.add_argument(
        "--cuslam-python",
        default="auto",
        help='Python command for cuVSLAM, or "auto" to detect one; e.g. "conda run -n cuvslam python"',
    )
    parser.add_argument(
        "--droid-python",
        default="auto",
        help='Python command for DROID-SLAM, or "auto" to detect one; e.g. "conda run -n droid-slam python"',
    )
    parser.add_argument(
        "--megasam-python",
        default="auto",
        help='Python command for MegaSAM, or "auto" to detect one; e.g. "conda run -n mega_sam python"',
    )
    parser.add_argument("--cuslam-extra-args", default="", help="extra args appended to cuVSLAM command")
    parser.add_argument("--droid-extra-args", default="", help="extra args appended to DROID-SLAM command")
    parser.add_argument("--megasam-extra-args", default="", help="extra args appended to MegaSAM command")

    parser.add_argument("--megasam-dir", help="directory containing MegaSAM EgoDex outputs")
    parser.add_argument("--megasam-tum", help="explicit MegaSAM relative OpenCV TUM file")
    parser.add_argument(
        "--megasam-variant",
        choices=("raw", "sim3", "auto"),
        default="sim3",
        help="which MegaSAM output to use when --megasam-tum is not set",
    )

    parser.add_argument("--droid-dir", help="directory containing DROID-SLAM EgoDex outputs")
    parser.add_argument("--droid-tum", help="explicit DROID-SLAM relative OpenCV TUM file")

    parser.add_argument("--cuslam-dir", help="directory containing cuVSLAM EgoDex outputs")
    parser.add_argument("--cuslam-tum", help="explicit cuVSLAM TUM file")
    parser.add_argument(
        "--cuslam-format",
        choices=("arkit-world", "relative-opencv"),
        default="arkit-world",
        help="format of --cuslam-tum/default cuVSLAM file",
    )
    parser.add_argument(
        "--prefer-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prefer scale-only outputs for monocular cuVSLAM/DROID when available",
    )
    parser.add_argument(
        "--skip-evo-aligned-pose",
        action="store_true",
        help="skip the final evo Sim(3)-aligned pose comparison plot",
    )

    parser.add_argument("--no-rerun", action="store_true", help="only write converted TUM/report files")
    parser.add_argument("--rerun-app-id", default="egodex_all_slam_comparison")
    parser.add_argument("--rerun-spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rrd", help="save a .rrd recording instead of spawning the viewer")
    parser.add_argument(
        "--save-rrd-default",
        action="store_true",
        help="save <output>/all_slam_comparison.rrd instead of spawning the viewer",
    )
    parser.add_argument("--rerun-axis-length", type=float, default=0.02)
    parser.add_argument("--rerun-trajectory-radius", type=float, default=0.00035)
    parser.add_argument("--rerun-origin-radius", type=float, default=0.0006)
    parser.add_argument(
        "--rerun-trajectory-mode",
        choices=("selected", "sim3", "both"),
        default="sim3",
        help="which trajectories to show in the final Rerun scene; sim3 uses the final auto-aligned paths",
    )
    parser.add_argument("--rerun-frame-stride", type=int, default=1, help="log every Nth timeline frame")
    parser.add_argument("--max-rerun-frames", type=int, help="debug cap for logged timeline frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rerun_frame_stride < 1:
        raise ValueError("--rerun-frame-stride must be >= 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    if args.rerun_axis_length <= 0:
        raise ValueError("--rerun-axis-length must be > 0")
    if args.rerun_origin_radius <= 0:
        raise ValueError("--rerun-origin-radius must be > 0")

    hdf5_path = resolve_path(args.hdf5)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing EgoDex hdf5: {hdf5_path}")
    mp4_path = resolve_path(args.mp4) if args.mp4 else paired_mp4_path(hdf5_path)
    if not mp4_path.exists():
        if args.run_slam:
            raise FileNotFoundError(f"Missing EgoDex mp4: {mp4_path}")
        print(f"Video not found; image stream will be skipped: {mp4_path}")
        mp4_path_or_none: Path | None = None
    else:
        mp4_path_or_none = mp4_path

    scene_name = sanitize_scene_name(args.scene_name) if args.scene_name else default_scene_name(hdf5_path)
    output_dir = resolve_path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    video_info = read_video_info(mp4_path_or_none)
    fps = args.fps or (float(video_info["fps"]) if video_info and video_info["fps"] else 30.0)

    print(f"Scene: {scene_name}")
    print(f"Output: {output_dir}")
    print(f"FPS: {fps:.6f}")

    world_from_camera_arkit = load_egodex_poses(hdf5_path)
    if args.end_frame is not None and args.end_frame <= args.start_frame:
        raise ValueError("--end-frame must be larger than --start-frame")

    if args.run_slam:
        assert mp4_path_or_none is not None
        output_dirs = run_slam_pipelines(
            args,
            hdf5_path=hdf5_path,
            mp4_path=mp4_path_or_none,
            scene_name=scene_name,
            output_dir=output_dir,
            fps=fps,
        )
        if args.dry_run:
            print("Dry run complete; skipping final aggregation.")
            return
        if not args.megasam_dir and not args.megasam_tum:
            args.megasam_dir = str(output_dirs["megasam"])
        if not args.droid_dir and not args.droid_tum:
            args.droid_dir = str(output_dirs["droid"])
        if not args.cuslam_dir and not args.cuslam_tum:
            args.cuslam_dir = str(output_dirs["cuslam"])
    else:
        run_root = output_dir / "runs"
        if not args.megasam_dir and not args.megasam_tum and (run_root / "megasam").exists():
            args.megasam_dir = str(run_root / "megasam")
        if not args.droid_dir and not args.droid_tum and (run_root / "droid").exists():
            args.droid_dir = str(run_root / "droid")
        if not args.cuslam_dir and not args.cuslam_tum and (run_root / "cuslam").exists():
            args.cuslam_dir = str(run_root / "cuslam")

    raw_paths: dict[str, tuple[Path | None, str]] = {}
    if not args.skip_megasam:
        raw_paths["megasam"] = select_megasam_path(args, scene_name)
    if not args.skip_droid:
        raw_paths["droid"] = select_droid_path(args, scene_name)
    if not args.skip_cuslam:
        raw_paths["cuslam"] = select_cuslam_path(args)

    missing: list[str] = []
    warnings: list[str] = []
    loaded_raw_rows: dict[str, np.ndarray] = {}
    for key, (path, search_note) in raw_paths.items():
        if path is None or not path.exists():
            missing.append(f"{key}: no TUM found ({search_note})")
            continue
        loaded_raw_rows[key] = load_tum(path)
        print(f"Loaded {key}: {path} ({len(loaded_raw_rows[key])} poses)")

    if not loaded_raw_rows:
        raise FileNotFoundError("No SLAM trajectory outputs were found.")

    origin_frame_id, origin_warnings = choose_origin_frame(
        list(loaded_raw_rows.values()),
        fps,
        args.origin_frame,
    )
    warnings.extend(origin_warnings)
    if origin_frame_id < 0 or origin_frame_id >= len(world_from_camera_arkit):
        raise ValueError(
            f"origin frame {origin_frame_id} is outside EgoDex GT range 0..{len(world_from_camera_arkit) - 1}"
        )
    print(f"Origin frame: {origin_frame_id}")

    max_gt_frame = len(world_from_camera_arkit) - 1
    if video_info is not None:
        max_gt_frame = min(max_gt_frame, int(video_info["frames"]) - 1)
    end_frame = args.end_frame if args.end_frame is not None else max_gt_frame + 1
    end_frame = min(end_frame, max_gt_frame + 1)
    gt_frame_ids = list(range(args.start_frame, end_frame, args.stride))
    if args.max_frames is not None:
        gt_frame_ids = gt_frame_ids[: args.max_frames]
    if not gt_frame_ids:
        raise ValueError("No GT frames selected; check --start-frame/--end-frame/--stride/--max-frames")
    gt_rows = relative_gt_rows(world_from_camera_arkit, fps, origin_frame_id, gt_frame_ids)
    gt_path = save_tum(output_dir / "gt_relative_opencv_tum.txt", gt_rows)

    trajectories: list[Trajectory] = []
    converted_paths: dict[str, str] = {}

    if "cuslam" in loaded_raw_rows:
        cuslam_source_path = raw_paths["cuslam"][0]
        assert cuslam_source_path is not None
        if args.cuslam_format == "arkit-world":
            cuslam_rows = arkit_world_rows_to_relative_opencv(
                loaded_raw_rows["cuslam"],
                world_from_camera_arkit[origin_frame_id],
            )
            note = "converted from cuVSLAM ARKit world TUM to relative OpenCV"
        else:
            cuslam_rows, warning = rebase_relative_rows(loaded_raw_rows["cuslam"], fps, origin_frame_id)
            note = "loaded as relative OpenCV TUM"
            if warning:
                warnings.append(f"cuVSLAM: {warning}")
                note += f"; {warning}"
        path = save_tum(output_dir / "cuslam_relative_opencv_tum.txt", cuslam_rows)
        converted_paths["cuslam"] = str(path)
        trajectories.append(Trajectory("cuslam", "cuVSLAM", cuslam_rows, CUSLAM_COLOR, cuslam_source_path, note))

    if "droid" in loaded_raw_rows:
        droid_source_path = raw_paths["droid"][0]
        assert droid_source_path is not None
        droid_rows, warning = rebase_relative_rows(loaded_raw_rows["droid"], fps, origin_frame_id)
        note = "relative OpenCV TUM"
        if warning:
            warnings.append(f"DROID-SLAM: {warning}")
            note += f"; {warning}"
        path = save_tum(output_dir / "droid_relative_opencv_tum.txt", droid_rows)
        converted_paths["droid"] = str(path)
        trajectories.append(Trajectory("droid", "DROID-SLAM", droid_rows, DROID_COLOR, droid_source_path, note))

    if "megasam" in loaded_raw_rows:
        megasam_source_path = raw_paths["megasam"][0]
        assert megasam_source_path is not None
        note = "relative OpenCV TUM"
        megasam_orientation_valid = True
        megasam_output_name = "megasam_relative_opencv_tum.txt"
        if "sim3" in megasam_source_path.name:
            megasam_rows = np.asarray(loaded_raw_rows["megasam"], dtype=np.float64).copy()
            warning = None
            note += "; GT-derived Sim(3) aligned MegaSAM positions"
            raw_orientation_path = megasam_source_path.parent / "megasam_relative_opencv_tum.txt"
            if raw_orientation_path.exists():
                raw_orientation_rows = load_tum(raw_orientation_path)
                megasam_rows, copied = copy_orientations_by_frame_id(
                    megasam_rows,
                    raw_orientation_rows,
                    fps,
                )
                if copied:
                    note += f"; orientations preserved from raw MegaSAM ({copied} poses)"
                else:
                    megasam_orientation_valid = False
                    note += "; raw MegaSAM orientations were not timestamp-matched"
            else:
                megasam_orientation_valid = False
                note += "; raw MegaSAM orientation file not found"
            megasam_output_name = "megasam_relative_opencv_sim3_tum.txt"
        else:
            megasam_rows, warning = rebase_relative_rows(loaded_raw_rows["megasam"], fps, origin_frame_id)
        if warning:
            warnings.append(f"MegaSAM: {warning}")
            note += f"; {warning}"
        path = save_tum(output_dir / megasam_output_name, megasam_rows)
        converted_paths["megasam"] = str(path)
        trajectories.append(
            Trajectory(
                "megasam",
                "MegaSAM",
                megasam_rows,
                MEGASAM_COLOR,
                megasam_source_path,
                note,
                orientation_valid=megasam_orientation_valid,
            )
        )

    stats = {
        trajectory.key: trajectory_stats(
            gt_rows,
            trajectory.rows,
            fps,
            include_rotation=trajectory.orientation_valid,
        )
        for trajectory in trajectories
    }
    evo_aligned = save_evo_aligned_pose_plots(
        gt_path,
        trajectories,
        converted_paths,
        output_dir,
        fps,
        skip=args.skip_evo_aligned_pose,
    )

    report_path = write_report(
        output_dir,
        scene_name=scene_name,
        hdf5_path=hdf5_path,
        mp4_path=mp4_path_or_none,
        fps=fps,
        origin_frame_id=origin_frame_id,
        gt_path=gt_path,
        trajectories=trajectories,
        stats=stats,
        missing=missing,
        warnings=warnings,
        converted_paths=converted_paths,
        evo_aligned=evo_aligned,
        rerun_trajectory_mode=args.rerun_trajectory_mode,
    )

    print(f"Saved GT TUM: {gt_path}")
    for trajectory in trajectories:
        values = stats.get(trajectory.key)
        if values is None:
            print(f"{trajectory.label}: no timestamp matches with GT")
        else:
            line = (
                f"{trajectory.label}: {values['frames']} matched poses | "
                f"translation RMSE {values['translation_rmse']:.6f} m"
            )
            if "rotation_rmse" in values:
                line += f" | rotation RMSE {values['rotation_rmse']:.3f} deg"
            else:
                line += " | rotation not meaningful after similarity alignment"
            print(line)
    if missing:
        print("Missing trajectories:")
        for item in missing:
            print(f"  - {item}")
    if warnings:
        print("Warnings:")
        for item in warnings:
            print(f"  - {item}")
    if evo_aligned.get("plots", {}).get("pose_comparison"):
        print(f"Saved evo aligned pose plot: {evo_aligned['plots']['pose_comparison']}")
    print(f"Saved report: {report_path}")

    display_trajectories = build_rerun_trajectories(
        trajectories,
        evo_aligned,
        args.rerun_trajectory_mode,
    )
    if args.rerun_trajectory_mode != "selected":
        print(f"Rerun trajectory mode: {args.rerun_trajectory_mode} ({len(display_trajectories)} displayed estimates)")

    visualize_in_rerun(
        gt_rows,
        display_trajectories,
        mp4_path_or_none,
        fps,
        args,
        output_dir,
    )


if __name__ == "__main__":
    main()
