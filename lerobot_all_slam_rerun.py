#!/usr/bin/env python3
"""Run cuVSLAM, MAC-VO, Mega-SAM and ORB-SLAM3 on LeRobot v3 stereo data and compare them in Rerun."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot_v3_common import (
    DEFAULT_CAMERA_PARAMS_KEY,
    DEFAULT_DATASET_ROOT,
    compute_rectified_calibration,
    DEFAULT_GT_COLUMN,
    DEFAULT_LEFT_KEY,
    DEFAULT_RIGHT_KEY,
    build_lerobot_manifest,
    default_scene_name,
    iter_video_frames,
    load_json,
    load_manifest,
    match_by_timestamp,
    load_tum,
    log_pose,
    log_static_trajectory,
    rr_image,
    rows_by_time_key,
    save_tum,
    save_manifest,
    set_rerun_time,
    trajectory_points,
    trajectory_stats,
    matrix_to_tum_row,
    tum_row_to_matrix,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tasks"
DEFAULT_EPISODE_INDEX = 0
DEFAULT_CUVSLAM_PYTHON = str(Path.home() / "miniconda3" / "envs" / "cuvslam" / "bin" / "python")
DEFAULT_MACVO_PYTHON = str(Path.home() / "miniconda3" / "envs" / "macvo" / "bin" / "python")
DEFAULT_MEGASAM_PYTHON = str(Path.home() / "miniconda3" / "envs" / "mega_sam" / "bin" / "python")
DEFAULT_ORBSLAM_ROOT = REPO_ROOT / "ORB_SLAM3"
DEFAULT_PREBUILT_MANIFEST_SOURCE = (
    REPO_ROOT / "cuVSLAM" / "examples" / "lerobotv3" / "outputs" / "episode_000000_head_raw_rect"
)
DEFAULT_PREBUILT_CUVSLAM_TUM = DEFAULT_PREBUILT_MANIFEST_SOURCE / "cuvslam_stereo_opencv_tum.txt"

GT_COLOR = [0, 200, 255]
CUVSLAM_COLOR = [255, 165, 0]
MACVO_COLOR = [255, 90, 90]
MEGASAM_COLOR = [120, 220, 120]
MEGASAM_MONO_UNIDEPTH_COLOR = [60, 255, 180]
ORBSLAM_COLOR = [180, 120, 255]


@dataclass
class Trajectory:
    key: str
    label: str
    rows: np.ndarray
    color: list[int]
    source_path: Path
    note: str


def build_manifest_from_prebuilt_source(
    *,
    source_dir: Path,
    dataset_root: Path,
    episode_index: int,
    left_key: str,
    right_key: str,
    camera_params_key: str,
    start_frame: int,
    end_frame: int | None,
    stride: int,
    max_frames: int | None,
    stereo_t_scale: float,
) -> dict[str, Any]:
    report_path = source_dir / "comparison_report.json"
    frame_ids_path = source_dir / "frame_ids.txt"
    timestamps_path = source_dir / "timestamps.txt"
    gt_path = source_dir / "gt_relative_opencv_tum.txt"
    if not report_path.exists() or not frame_ids_path.exists() or not timestamps_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Prebuilt manifest source is incomplete: {source_dir}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    full_frame_ids = np.loadtxt(frame_ids_path, dtype=np.int64)
    full_timestamps = np.loadtxt(timestamps_path, dtype=np.float64)
    full_gt_rows = load_tum(gt_path)
    if full_frame_ids.ndim == 0:
        full_frame_ids = np.asarray([int(full_frame_ids)], dtype=np.int64)
    if full_timestamps.ndim == 0:
        full_timestamps = np.asarray([float(full_timestamps)], dtype=np.float64)

    end_value = len(full_frame_ids) if end_frame is None else min(int(end_frame), len(full_frame_ids))
    selected_indices = list(range(int(start_frame), end_value, int(stride)))
    if max_frames is not None:
        selected_indices = selected_indices[: int(max_frames)]
    if not selected_indices:
        raise ValueError("No frames selected from prebuilt episode source")

    frame_ids = full_frame_ids[selected_indices]
    timestamps = full_timestamps[selected_indices]
    gt_rows = full_gt_rows[selected_indices]

    info = load_json(dataset_root / "meta" / "info.json")
    calibration = compute_rectified_calibration(
        info,
        camera_params_key=camera_params_key,
        image_key=left_key,
        stereo_t_scale=stereo_t_scale,
    )

    raw_left_path = report["left_video"]
    raw_right_path = report["right_video"]
    left_video = str(raw_left_path).replace(report["left_key"], left_key)
    right_video = str(raw_right_path).replace(report["right_key"], right_key)
    scene_name = default_scene_name(int(episode_index), left_key)

    return {
        "dataset_root": str(dataset_root),
        "episode_index": int(episode_index),
        "episode_length": int(report.get("episode_length", len(full_frame_ids))),
        "fps": float(report["fps"]),
        "scene_name": scene_name,
        "left_key": left_key,
        "right_key": right_key,
        "camera_params_key": camera_params_key,
        "gt_column": report["gt_column"],
        "gt_column_requested": report.get("gt_column_requested", report["gt_column"]),
        "gt_euler_order": report["gt_euler_order"],
        "gt_source_frame": report["gt_source_frame"],
        "gt_pose_convention": report["gt_pose_convention"],
        "frame_ids": frame_ids.astype(int).tolist(),
        "timestamps": timestamps.astype(float).tolist(),
        "left_video": {
            "key": left_key,
            "path": left_video,
            "from_timestamp": 0.0,
            "to_timestamp": float(full_timestamps[-1]) if len(full_timestamps) else 0.0,
            "start_frame": int(report.get("left_video_start_frame", 0)),
            "duration_frames": int(report.get("episode_length", len(full_frame_ids))),
        },
        "right_video": {
            "key": right_key,
            "path": right_video,
            "from_timestamp": 0.0,
            "to_timestamp": float(full_timestamps[-1]) if len(full_timestamps) else 0.0,
            "start_frame": int(report.get("right_video_start_frame", 0)),
            "duration_frames": int(report.get("episode_length", len(full_frame_ids))),
        },
        "left_video_frame_ids": (int(report.get("left_video_start_frame", 0)) + frame_ids).astype(int).tolist(),
        "right_video_frame_ids": (int(report.get("right_video_start_frame", 0)) + frame_ids).astype(int).tolist(),
        "gt_rows": gt_rows.tolist(),
        "calibration": calibration,
        "coordinate_note": "All trajectories use first-selected-frame-relative OpenCV optical camera poses (x right, y down, z forward).",
        "prebuilt_source": str(source_dir),
    }


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
    if override and python_command_works(override, import_check, cwd):
        print(f"Auto-selected {label} Python from ${env_var}: {override}")
        return override

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
        f"Pass --{label}-python explicitly."
    )


def run_command(command: list[str], *, cwd: Path, dry_run: bool, continue_on_error: bool) -> bool:
    print("$ " + " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return True
    try:
        subprocess.run(command, cwd=str(cwd), check=True)
        return True
    except subprocess.CalledProcessError as exc:
        if not continue_on_error:
            raise
        print(f"Command failed with exit code {exc.returncode}; continuing.")
        return False


def expected_output_ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def estimate_scale_to_reference(
    reference_rows: np.ndarray,
    estimate_rows: np.ndarray,
    max_time_diff: float,
    *,
    method: str,
    min_displacement: float,
) -> dict[str, Any]:
    ref, est = match_by_timestamp(reference_rows, estimate_rows, max_time_diff)
    if len(ref) < 2:
        raise ValueError("not enough timestamp matches to estimate scale")

    finite = np.isfinite(ref[:, 1:4]).all(axis=1) & np.isfinite(est[:, 1:4]).all(axis=1)
    ref = ref[finite]
    est = est[finite]
    if len(ref) < 2:
        raise ValueError("not enough finite matched positions to estimate scale")

    reference_origin = ref[0, 1:4].copy()
    estimate_origin = est[0, 1:4].copy()
    min_displacement = max(float(min_displacement), 0.0)
    if method == "rms":
        ref_offsets = ref[:, 1:4] - reference_origin
        est_offsets = est[:, 1:4] - estimate_origin
        ref_distances = np.linalg.norm(ref_offsets, axis=1)
        est_distances = np.linalg.norm(est_offsets, axis=1)
        valid = (ref_distances > min_displacement) & (est_distances > min_displacement)
        if int(np.count_nonzero(valid)) < 2:
            raise ValueError("not enough non-zero matched displacements to estimate RMS scale")
        reference_measure = float(np.sqrt(np.mean(ref_distances[valid] ** 2)))
        estimate_measure = float(np.sqrt(np.mean(est_distances[valid] ** 2)))
        used = int(np.count_nonzero(valid))
    elif method == "path":
        ref_steps = np.linalg.norm(np.diff(ref[:, 1:4], axis=0), axis=1)
        est_steps = np.linalg.norm(np.diff(est[:, 1:4], axis=0), axis=1)
        valid = (ref_steps > min_displacement) & (est_steps > min_displacement)
        if int(np.count_nonzero(valid)) < 1:
            raise ValueError("not enough non-zero matched steps to estimate path scale")
        reference_measure = float(np.sum(ref_steps[valid]))
        estimate_measure = float(np.sum(est_steps[valid]))
        used = int(np.count_nonzero(valid))
    else:
        raise ValueError(f"Unsupported scale alignment method: {method}")

    if estimate_measure <= 1e-12:
        raise ValueError("estimated trajectory displacement is too small to scale")
    scale = reference_measure / estimate_measure
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid scale estimate: {scale}")

    return {
        "type": "scale_translation",
        "method": method,
        "scale": float(scale),
        "matched_frames": int(len(ref)),
        "used_samples": used,
        "reference_measure_m": reference_measure,
        "estimate_measure_m": estimate_measure,
        "origin_timestamp": float(est[0, 0]),
        "reference_origin_xyz": reference_origin.tolist(),
        "estimate_origin_xyz": estimate_origin.tolist(),
    }


def apply_scale_to_rows(
    rows: np.ndarray,
    scale: float,
    *,
    estimate_origin: list[float] | np.ndarray | None = None,
    reference_origin: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    scaled = np.asarray(rows, dtype=np.float64).copy()
    source_origin = (
        scaled[0, 1:4].copy()
        if estimate_origin is None
        else np.asarray(estimate_origin, dtype=np.float64)
    )
    target_origin = source_origin if reference_origin is None else np.asarray(reference_origin, dtype=np.float64)
    scaled[:, 1:4] = target_origin + float(scale) * (scaled[:, 1:4] - source_origin)
    return scaled


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
        "type": "sim3",
        "method": "umeyama",
        "scale": scale,
        "rotation_matrix": rotation.tolist(),
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


def resolve_alignment_reference(
    *,
    reference_key: str,
    gt_rows: np.ndarray,
    gt_tum_path: Path,
    trajectories: list[Trajectory],
) -> tuple[str, str, np.ndarray, Path]:
    by_key = {trajectory.key: trajectory for trajectory in trajectories}
    if reference_key == "auto":
        for key in ("megasam", "macvo", "orbslam", "cuvslam"):
            if key in by_key:
                trajectory = by_key[key]
                return trajectory.key, trajectory.label, trajectory.rows, trajectory.source_path
        return "gt", "GT", gt_rows, gt_tum_path
    if reference_key == "gt":
        return "gt", "GT", gt_rows, gt_tum_path
    if reference_key not in by_key:
        available = ", ".join(["gt", *by_key.keys()])
        raise ValueError(f"Alignment reference '{reference_key}' is unavailable. Available: {available}")
    trajectory = by_key[reference_key]
    return trajectory.key, trajectory.label, trajectory.rows, trajectory.source_path


def parse_key_list(text: str | None) -> set[str]:
    if not text:
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def option_value_from_extra_args(parts: list[str], name: str, default: str) -> str:
    prefix = f"{name}="
    for index, part in enumerate(parts):
        if part == name and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith(prefix):
            return part[len(prefix) :]
    return default


def megasam_label_and_note(extra_args: str) -> tuple[str, str]:
    parts = shlex.split(extra_args)
    metric_depth_source = option_value_from_extra_args(parts, "--metric-depth-source", "stereo")
    depth_source = option_value_from_extra_args(parts, "--depth-source", "metric")
    if metric_depth_source == "unidepth":
        if depth_source == "mono_aligned":
            return (
                "Mega-SAM Mono (UniDepth+DA)",
                "treats LeRobot stereo as a monocular left-camera video with Depth-Anything aligned to UniDepth",
            )
        return (
            "Mega-SAM Mono (UniDepth)",
            "treats LeRobot stereo as a monocular left-camera video with UniDepth metric depth",
        )
    if depth_source == "mono_aligned":
        return (
            "Mega-SAM Mono (DA+StereoDepth)",
            "mono tracking on left images with Depth-Anything aligned to stereo metric depth",
        )
    return (
        "Mega-SAM Mono",
        "mono tracking on left images with metric stereo depth from the right camera",
    )


def repeated_pose_tail_info(
    rows: np.ndarray,
    *,
    check_frames: int,
    static_step_threshold: float,
    min_static_steps: int,
) -> dict[str, Any] | None:
    rows = np.asarray(rows, dtype=np.float64)
    if len(rows) < 3:
        return None
    steps = np.linalg.norm(np.diff(rows[:, 1:4], axis=0), axis=1)
    tail_start = max(0, len(steps) - max(int(check_frames), 1))
    static_indices = np.flatnonzero(steps[tail_start:] <= float(static_step_threshold)) + tail_start
    if len(static_indices) < int(min_static_steps):
        return None

    first_static_step = int(static_indices[0])
    keep_rows = first_static_step + 1
    dropped_rows = len(rows) - keep_rows
    if dropped_rows <= 0:
        return None
    return {
        "type": "repeated_pose_tail",
        "check_frames": int(check_frames),
        "static_step_threshold_m": float(static_step_threshold),
        "min_static_steps": int(min_static_steps),
        "static_step_indices": static_indices.astype(int).tolist(),
        "first_dropped_index": int(keep_rows),
        "kept_frames": int(keep_rows),
        "dropped_frames": int(dropped_rows),
        "original_frames": int(len(rows)),
    }


def trim_megasam_tail_for_display(
    output_dir: Path,
    trajectories: list[Trajectory],
    *,
    enabled: bool,
    check_frames: int,
    static_step_threshold: float,
    min_static_steps: int,
) -> tuple[list[Trajectory], dict[str, Any], list[str]]:
    trim_info: dict[str, Any] = {
        "enabled": bool(enabled),
        "method": "repeated_pose_tail",
        "trajectories": {},
    }
    if not enabled:
        return trajectories, trim_info, []

    trimmed_dir = output_dir / "tail_trimmed"
    result: list[Trajectory] = []
    warnings: list[str] = []
    for trajectory in trajectories:
        if not trajectory.key.startswith("megasam"):
            result.append(trajectory)
            continue

        info = repeated_pose_tail_info(
            trajectory.rows,
            check_frames=check_frames,
            static_step_threshold=static_step_threshold,
            min_static_steps=min_static_steps,
        )
        if info is None:
            trim_info["trajectories"][trajectory.key] = {
                "source_tum": str(trajectory.source_path),
                "trimmed": False,
            }
            result.append(trajectory)
            continue

        trimmed_rows = trajectory.rows[: int(info["kept_frames"])]
        trimmed_path = save_tum(trimmed_dir / f"{trajectory.key}_tail_trimmed_tum.txt", trimmed_rows)
        info.update(
            source_tum=str(trajectory.source_path),
            trimmed_tum=str(trimmed_path),
        )
        trim_info["trajectories"][trajectory.key] = info
        warnings.append(
            f"{trajectory.label}: dropped {info['dropped_frames']} repeated-pose tail frames "
            f"for display/scale alignment ({trajectory.source_path})"
        )
        result.append(
            Trajectory(
                key=trajectory.key,
                label=trajectory.label,
                rows=trimmed_rows,
                color=trajectory.color,
                source_path=trimmed_path,
                note=(
                    f"{trajectory.note}; dropped {info['dropped_frames']} repeated-pose tail frames "
                    f"from raw source {trajectory.source_path}"
                ),
            )
        )

    return result, trim_info, warnings


def build_scale_aligned_trajectories(
    output_dir: Path,
    gt_rows: np.ndarray,
    trajectories: list[Trajectory],
    *,
    reference_path: Path,
    max_time_diff: float,
    method: str,
    min_displacement: float,
    skip_keys: set[str],
) -> tuple[list[Trajectory], dict[str, Any], list[str]]:
    scale_dir = output_dir / "scale_aligned"
    scale_dir.mkdir(parents=True, exist_ok=True)
    display_trajectories: list[Trajectory] = []
    warnings: list[str] = []
    alignment: dict[str, Any] = {
        "enabled": True,
        "reference_tum": str(reference_path),
        "method": method,
        "min_displacement_m": float(min_displacement),
        "output_dir": str(scale_dir),
        "trajectories": {},
    }

    for trajectory in trajectories:
        if trajectory.key in skip_keys:
            alignment["trajectories"][trajectory.key] = {
                "source_tum": str(trajectory.source_path),
                "skipped": True,
                "reason": "raw metric trajectory kept for display",
            }
            display_trajectories.append(
                Trajectory(
                    key=trajectory.key,
                    label=trajectory.label,
                    rows=trajectory.rows,
                    color=trajectory.color,
                    source_path=trajectory.source_path,
                    note=f"{trajectory.note}; scale alignment skipped, raw metric trajectory kept for display",
                )
            )
            continue

        try:
            result = estimate_scale_to_reference(
                gt_rows,
                trajectory.rows,
                max_time_diff,
                method=method,
                min_displacement=min_displacement,
            )
        except ValueError as exc:
            message = f"{trajectory.key}: scale alignment failed ({exc}); using raw trajectory for display"
            warnings.append(message)
            alignment["trajectories"][trajectory.key] = {
                "source_tum": str(trajectory.source_path),
                "error": str(exc),
            }
            display_trajectories.append(trajectory)
            continue

        scaled_rows = apply_scale_to_rows(
            trajectory.rows,
            float(result["scale"]),
            estimate_origin=result.get("estimate_origin_xyz"),
            reference_origin=result.get("reference_origin_xyz"),
        )
        scaled_path = save_tum(scale_dir / f"{trajectory.key}_scale_aligned_tum.txt", scaled_rows)
        result.update(
            source_tum=str(trajectory.source_path),
            aligned_tum=str(scaled_path),
        )
        alignment["trajectories"][trajectory.key] = result
        display_trajectories.append(
            Trajectory(
                key=trajectory.key,
                label=trajectory.label,
                rows=scaled_rows,
                color=trajectory.color,
                source_path=scaled_path,
                note=(
                    f"scale/origin aligned to GT with scale={float(result['scale']):.9g}; "
                    f"raw source: {trajectory.source_path}"
                ),
            )
        )

    return display_trajectories, alignment, warnings


def build_sim3_aligned_trajectories(
    output_dir: Path,
    gt_rows: np.ndarray,
    trajectories: list[Trajectory],
    *,
    gt_tum_path: Path,
    reference_key: str,
    max_time_diff: float,
    min_displacement: float,
    skip_keys: set[str],
) -> tuple[list[Trajectory], dict[str, Any], list[str]]:
    sim3_dir = output_dir / "sim3_aligned"
    sim3_dir.mkdir(parents=True, exist_ok=True)
    ref_key, ref_label, ref_rows, ref_path = resolve_alignment_reference(
        reference_key=reference_key,
        gt_rows=gt_rows,
        gt_tum_path=gt_tum_path,
        trajectories=trajectories,
    )
    display_trajectories: list[Trajectory] = []
    warnings: list[str] = []
    alignment: dict[str, Any] = {
        "enabled": True,
        "mode": "sim3",
        "method": "umeyama",
        "reference_key": ref_key,
        "reference_label": ref_label,
        "reference_tum": str(ref_path),
        "min_displacement_m": float(min_displacement),
        "output_dir": str(sim3_dir),
        "trajectories": {},
    }

    for trajectory in trajectories:
        if trajectory.key == ref_key:
            alignment["trajectories"][trajectory.key] = {
                "source_tum": str(trajectory.source_path),
                "skipped": True,
                "reason": "alignment reference kept unchanged",
            }
            display_trajectories.append(
                Trajectory(
                    key=trajectory.key,
                    label=trajectory.label,
                    rows=trajectory.rows,
                    color=trajectory.color,
                    source_path=trajectory.source_path,
                    note=f"{trajectory.note}; Sim3 reference trajectory kept unchanged",
                )
            )
            continue

        if trajectory.key in skip_keys:
            alignment["trajectories"][trajectory.key] = {
                "source_tum": str(trajectory.source_path),
                "skipped": True,
                "reason": "raw metric trajectory kept for display",
            }
            display_trajectories.append(
                Trajectory(
                    key=trajectory.key,
                    label=trajectory.label,
                    rows=trajectory.rows,
                    color=trajectory.color,
                    source_path=trajectory.source_path,
                    note=f"{trajectory.note}; Sim3 alignment skipped, raw trajectory kept for display",
                )
            )
            continue

        try:
            result = estimate_sim3_to_reference(
                ref_rows,
                trajectory.rows,
                max_time_diff,
                min_displacement=min_displacement,
            )
        except ValueError as exc:
            message = f"{trajectory.key}: Sim3 alignment failed ({exc}); using raw trajectory for display"
            warnings.append(message)
            alignment["trajectories"][trajectory.key] = {
                "source_tum": str(trajectory.source_path),
                "error": str(exc),
            }
            display_trajectories.append(trajectory)
            continue

        aligned_rows = apply_sim3_to_rows(
            trajectory.rows,
            scale=float(result["scale"]),
            rotation_matrix=result["rotation_matrix"],
            translation_xyz=result["translation_xyz"],
        )
        aligned_path = save_tum(sim3_dir / f"{trajectory.key}_sim3_aligned_tum.txt", aligned_rows)
        result.update(
            source_tum=str(trajectory.source_path),
            aligned_tum=str(aligned_path),
            reference_key=ref_key,
            reference_tum=str(ref_path),
        )
        alignment["trajectories"][trajectory.key] = result
        display_trajectories.append(
            Trajectory(
                key=trajectory.key,
                label=trajectory.label,
                rows=aligned_rows,
                color=trajectory.color,
                source_path=aligned_path,
                note=(
                    f"Sim3 aligned to {ref_label} with scale={float(result['scale']):.9g}; "
                    f"raw source: {trajectory.source_path}"
                ),
            )
        )

    return display_trajectories, alignment, warnings


def stats_line(values: dict[str, float | int] | None) -> str:
    if values is None:
        return "stats=None"
    return (
        f"frames={values['frames']} "
        f"trans_rmse={values['translation_rmse']:.6f}m "
        f"trans_mean={values['translation_mean']:.6f}m "
        f"trans_max={values['translation_max']:.6f}m "
        f"rot_rmse={values['rotation_rmse']:.3f}deg "
        f"rot_mean={values['rotation_mean']:.3f}deg "
        f"rot_max={values['rotation_max']:.3f}deg"
    )


def init_rerun(args: argparse.Namespace, output_dir: Path) -> Any:
    try:
        import rerun as rr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing rerun-sdk. Install it in the environment running this script, "
            "or pass --no-rerun."
        ) from exc

    save_rrd = Path(args.save_rrd).expanduser().resolve() if args.save_rrd else None
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
    except Exception as exc:  # pragma: no cover
        print(f"Rerun coordinate setup failed, using defaults: {exc}")

    try:
        import rerun.blueprint as rrb

        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Horizontal(
                    column_shares=[0.42, 0.58],
                    contents=[
                        rrb.Spatial2DView(name="Left Image", origin="world/input/left"),
                        rrb.Spatial3DView(name="GT / cuVSLAM / MAC-VO / Mega-SAM / ORB-SLAM3", origin="world"),
                    ],
                ),
            ),
            make_active=True,
        )
    except Exception as exc:  # pragma: no cover
        print(f"Rerun blueprint setup failed, using default layout: {exc}")
    return rr


def visualize_in_rerun(
    manifest: dict[str, Any],
    gt_rows: np.ndarray,
    trajectories: list[Trajectory],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    if args.no_rerun:
        return

    rr = init_rerun(args, output_dir)
    log_static_trajectory(rr, "world/gt", gt_rows, GT_COLOR, args.rerun_trajectory_radius)
    for trajectory in trajectories:
        log_static_trajectory(rr, f"world/{trajectory.key}", trajectory.rows, trajectory.color, args.rerun_trajectory_radius)

    gt_lookup = rows_by_time_key(gt_rows)
    lookups = {trajectory.key: rows_by_time_key(trajectory.rows) for trajectory in trajectories}

    left_video_path = Path(manifest["left_video"]["path"])
    frame_ids = manifest["frame_ids"]
    timestamps = manifest["timestamps"]
    left_video_frame_ids = manifest["left_video_frame_ids"]
    if args.max_rerun_frames is not None:
        frame_ids = frame_ids[: args.max_rerun_frames]
        timestamps = timestamps[: args.max_rerun_frames]
        left_video_frame_ids = left_video_frame_ids[: args.max_rerun_frames]

    frame_triplets = list(zip(frame_ids, timestamps, left_video_frame_ids))
    if args.rerun_frame_stride > 1:
        frame_triplets = [
            triplet for index, triplet in enumerate(frame_triplets) if index % args.rerun_frame_stride == 0
        ]

    image_frames: dict[int, np.ndarray] = {}
    if frame_triplets:
        try:
            image_frames = {
                int(video_frame_id): image_rgb
                for video_frame_id, image_rgb in iter_video_frames(
                    left_video_path,
                    [int(video_frame_id) for _, _, video_frame_id in frame_triplets],
                    float(manifest["fps"]),
                )
            }
        except Exception as exc:
            print(f"Cannot decode left video for Rerun images; showing trajectories only: {exc}")

    logged = 0
    for frame_id, timestamp, video_frame_id in frame_triplets:
        set_rerun_time(rr, int(frame_id), float(timestamp))
        image_rgb = image_frames.get(int(video_frame_id))
        if image_rgb is not None:
            rr.log("world/input/left", rr_image(rr, image_rgb))

        row_key = int(round(float(timestamp) * 1_000_000.0))
        log_pose(rr, "world/gt/current", gt_lookup.get(row_key), GT_COLOR, args.rerun_axis_length, args.rerun_origin_radius)
        for trajectory in trajectories:
            log_pose(
                rr,
                f"world/{trajectory.key}/current",
                lookups[trajectory.key].get(row_key),
                trajectory.color,
                args.rerun_axis_length,
                args.rerun_origin_radius,
            )
        logged += 1

    print(f"Sent Rerun visualization: {logged} frames")
    if args.save_rrd:
        print(f"Saved Rerun recording: {args.save_rrd}")


def write_report(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    trajectories: list[Trajectory],
    raw_stats: dict[str, dict[str, float | int] | None],
    display_trajectories: list[Trajectory],
    display_stats: dict[str, dict[str, float | int] | None],
    tail_trim: dict[str, Any],
    scale_alignment: dict[str, Any],
    missing: list[str],
    warnings: list[str],
) -> Path:
    display_by_key = {trajectory.key: trajectory for trajectory in display_trajectories}
    report = {
        "scene_name": manifest["scene_name"],
        "dataset_root": manifest["dataset_root"],
        "episode_index": manifest["episode_index"],
        "fps": manifest["fps"],
        "coordinate_note": manifest["coordinate_note"],
        "tail_trim": tail_trim,
        "scale_alignment": scale_alignment,
        "trajectory_alignment": scale_alignment,
        "trajectories": {
            trajectory.key: {
                "label": trajectory.label,
                "source": str(trajectory.source_path),
                "note": trajectory.note,
                "stats_vs_gt": display_stats.get(trajectory.key),
                "stats_vs_gt_raw": raw_stats.get(trajectory.key),
                "stats_vs_gt_display": display_stats.get(trajectory.key),
                "display_source": str(display_by_key.get(trajectory.key, trajectory).source_path),
                "display_note": display_by_key.get(trajectory.key, trajectory).note,
            }
            for trajectory in trajectories
        },
        "missing": missing,
        "warnings": warnings,
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    text_path = output_dir / "comparison_report.txt"
    with text_path.open("w", encoding="utf-8") as file:
        file.write("LeRobot v3 stereo all-SLAM comparison\n")
        file.write(f"scene_name={manifest['scene_name']}\n")
        file.write(f"episode_index={manifest['episode_index']}\n")
        file.write(f"dataset_root={manifest['dataset_root']}\n")
        file.write(f"fps={manifest['fps']:.6f}\n")
        file.write("coordinate=first-selected-frame-relative OpenCV optical camera (x right, y down, z forward)\n\n")
        if scale_alignment.get("enabled"):
            file.write(
                "trajectory_alignment=enabled "
                f"mode={scale_alignment.get('mode', 'scale')} "
                f"method={scale_alignment.get('method')} "
                f"min_displacement_m={scale_alignment.get('min_displacement_m')}\n"
            )
            file.write(f"alignment_reference_key={scale_alignment.get('reference_key')}\n")
            file.write(f"alignment_reference={scale_alignment.get('reference_tum')}\n")
            file.write(f"aligned_dir={scale_alignment.get('output_dir')}\n\n")
        else:
            file.write("trajectory_alignment=disabled\n\n")
        if tail_trim.get("enabled"):
            file.write("tail_trim=enabled method=repeated_pose_tail\n\n")
        else:
            file.write("tail_trim=disabled\n\n")

        for trajectory in trajectories:
            raw_values = raw_stats.get(trajectory.key)
            display_values = display_stats.get(trajectory.key)
            trajectory_trim = tail_trim.get("trajectories", {}).get(trajectory.key, {})
            trajectory_alignment = scale_alignment.get("trajectories", {}).get(trajectory.key, {})
            file.write(f"[{trajectory.label}]\n")
            file.write(f"source={trajectory.source_path}\n")
            file.write(f"note={trajectory.note}\n")
            file.write(f"raw {stats_line(raw_values)}\n")
            if trajectory_trim.get("trimmed_tum"):
                file.write(
                    "tail_trim="
                    f"type={trajectory_trim.get('type')} "
                    f"dropped_frames={trajectory_trim.get('dropped_frames')} "
                    f"kept_frames={trajectory_trim.get('kept_frames')} "
                    f"first_dropped_index={trajectory_trim.get('first_dropped_index')}\n"
                )
                file.write(f"tail_trimmed_tum={trajectory_trim['trimmed_tum']}\n")
            if trajectory_alignment.get("aligned_tum"):
                file.write(
                    "trajectory_alignment="
                    f"type={trajectory_alignment.get('type')} "
                    f"method={trajectory_alignment.get('method')} "
                    f"scale={float(trajectory_alignment['scale']):.9g} "
                    f"matched_frames={trajectory_alignment.get('matched_frames')} "
                    f"used_samples={trajectory_alignment.get('used_samples')}\n"
                )
                file.write(f"aligned_tum={trajectory_alignment['aligned_tum']}\n")
                if trajectory_alignment.get("residual_rmse_m") is not None:
                    file.write(
                        "alignment_residual="
                        f"rmse={float(trajectory_alignment['residual_rmse_m']):.6f}m "
                        f"mean={float(trajectory_alignment['residual_mean_m']):.6f}m "
                        f"max={float(trajectory_alignment['residual_max_m']):.6f}m\n"
                    )
                file.write(f"aligned_display {stats_line(display_values)}\n\n")
            elif trajectory_alignment.get("skipped"):
                file.write(f"trajectory_alignment=skipped reason={trajectory_alignment.get('reason')}\n")
                file.write(f"display {stats_line(display_values)}\n\n")
            elif trajectory_alignment.get("error"):
                file.write(f"trajectory_alignment_error={trajectory_alignment['error']}\n")
                file.write(f"display {stats_line(display_values)}\n\n")
            else:
                file.write(f"display {stats_line(display_values)}\n\n")
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
    return text_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cuVSLAM stereo, MAC-VO stereo, Mega-SAM mono and ORB-SLAM3 stereo on LeRobot v3, then compare them in Rerun."
    )
    parser.add_argument("--manifest", type=Path, help="reuse an existing manifest.json instead of rebuilding it from LeRobot parquet metadata")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode-index", type=int, default=DEFAULT_EPISODE_INDEX)
    parser.add_argument("--output", type=Path, help="default: tasks/<scene_name>")
    parser.add_argument("--left-key", default=DEFAULT_LEFT_KEY)
    parser.add_argument("--right-key", default=DEFAULT_RIGHT_KEY)
    parser.add_argument("--camera-params-key", default=DEFAULT_CAMERA_PARAMS_KEY)
    parser.add_argument("--gt-column", default=DEFAULT_GT_COLUMN)
    parser.add_argument("--gt-euler-order", default="xyz")
    parser.add_argument("--gt-source-frame", choices=("robot_base", "opencv"), default="robot_base")
    parser.add_argument("--gt-pose-convention", choices=("world_from_camera", "camera_from_world"), default="world_from_camera")
    parser.add_argument("--fps", type=float, help="override dataset/video FPS")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="episode-relative exclusive end frame")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stereo-t-scale", type=float, default=0.001)

    parser.add_argument("--run-slam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-cuvslam", action="store_true")
    parser.add_argument("--skip-macvo", action="store_true")
    parser.add_argument("--skip-megasam", action="store_true")
    parser.add_argument("--skip-orbslam", action="store_true")

    parser.add_argument("--cuvslam-python", default=DEFAULT_CUVSLAM_PYTHON)
    parser.add_argument("--macvo-python", default=DEFAULT_MACVO_PYTHON)
    parser.add_argument("--megasam-python", default=DEFAULT_MEGASAM_PYTHON)
    parser.add_argument("--orbslam-root", type=Path, default=DEFAULT_ORBSLAM_ROOT)
    parser.add_argument("--cuvslam-extra-args", default="")
    parser.add_argument("--macvo-extra-args", default="")
    parser.add_argument("--megasam-extra-args", default="")
    parser.add_argument("--orbslam-extra-args", default="")

    parser.add_argument(
        "--megasam-auto-trim-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop a repeated-pose suffix from Mega-SAM display/alignment outputs",
    )
    parser.add_argument("--megasam-tail-check-frames", type=int, default=24)
    parser.add_argument("--megasam-tail-static-step", type=float, default=1e-5)
    parser.add_argument("--megasam-tail-min-static-steps", type=int, default=3)

    parser.add_argument(
        "--scale-align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="align loaded SLAM trajectories before report/Rerun display; --no-scale-align disables all alignment",
    )
    parser.add_argument(
        "--alignment-mode",
        choices=("scale", "sim3"),
        default="sim3",
        help="sim3 applies scale+rotation+translation to a SLAM reference; scale keeps the older scale+origin alignment",
    )
    parser.add_argument(
        "--alignment-reference-key",
        default="auto",
        help="reference for Sim3 alignment: auto, gt, cuvslam, macvo, megasam, or orbslam; auto prefers Mega-SAM",
    )
    parser.add_argument(
        "--scale-align-method",
        choices=("rms", "path"),
        default="rms",
        help="scale mode only: rms matches RMS displacement from the first pose; path matches cumulative path length",
    )
    parser.add_argument(
        "--scale-align-min-displacement",
        type=float,
        default=0.0,
        help="ignore tiny matched displacements when estimating scale, in meters",
    )
    parser.add_argument(
        "--scale-align-skip-keys",
        default="",
        help="comma-separated trajectory keys to keep at raw scale during display alignment; default aligns all loaded SLAM trajectories",
    )

    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument("--rerun-app-id", default="lerobot_all_slam_comparison")
    parser.add_argument("--rerun-spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rrd", help="save a .rrd recording instead of spawning the viewer")
    parser.add_argument("--rerun-axis-length", type=float, default=0.04)
    parser.add_argument("--rerun-trajectory-radius", type=float, default=0.001)
    parser.add_argument("--rerun-origin-radius", type=float, default=0.004)
    parser.add_argument("--rerun-frame-stride", type=int, default=1)
    parser.add_argument("--max-rerun-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest is not None:
        manifest = load_manifest(args.manifest)
    elif (
        args.episode_index == DEFAULT_EPISODE_INDEX
        and args.dataset_root.expanduser().resolve() == DEFAULT_DATASET_ROOT.expanduser().resolve()
        and DEFAULT_PREBUILT_MANIFEST_SOURCE.exists()
    ):
        manifest = build_manifest_from_prebuilt_source(
            source_dir=DEFAULT_PREBUILT_MANIFEST_SOURCE,
            dataset_root=args.dataset_root.expanduser().resolve(),
            episode_index=args.episode_index,
            left_key=args.left_key,
            right_key=args.right_key,
            camera_params_key=args.camera_params_key,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            max_frames=args.max_frames,
            stereo_t_scale=args.stereo_t_scale,
        )
    else:
        manifest = build_lerobot_manifest(
            dataset_root=args.dataset_root,
            episode_index=args.episode_index,
            left_key=args.left_key,
            right_key=args.right_key,
            camera_params_key=args.camera_params_key,
            gt_column=args.gt_column,
            gt_euler_order=args.gt_euler_order,
            gt_source_frame=args.gt_source_frame,
            gt_pose_convention=args.gt_pose_convention,
            fps=args.fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            max_frames=args.max_frames,
            stereo_t_scale=args.stereo_t_scale,
        )

    scene_name = manifest["scene_name"] or default_scene_name(manifest["episode_index"], manifest["left_key"])
    output_dir = (args.output or (DEFAULT_OUTPUT_ROOT / scene_name)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, gt_tum_path, _ = save_manifest(output_dir, manifest)
    gt_rows = load_tum(gt_tum_path)

    print(f"Scene: {scene_name}")
    print(f"Output: {output_dir}")
    print(f"Left key: {args.left_key}")
    print(f"Right key: {args.right_key}")
    print(f"FPS: {manifest['fps']:.6f}")
    print(f"Selected frames: {len(manifest['frame_ids'])} ({manifest['frame_ids'][0]} -> {manifest['frame_ids'][-1]})")

    run_root = output_dir / "runs"
    output_dirs = {
        "cuvslam": run_root / "cuvslam",
        "macvo": run_root / "macvo",
        "megasam": run_root / "megasam",
        "orbslam": run_root / "orbslam",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    if args.run_slam:
        if not args.skip_cuvslam:
            cuvslam_python = resolve_python_command(
                args.cuvslam_python,
                label="cuvslam",
                env_var="LEROBOT_CUVSLAM_PYTHON",
                env_names=["cuvslam"],
                import_check="import cuvslam",
                cwd=REPO_ROOT,
            )
            expected = output_dirs["cuvslam"] / "cuvslam_stereo_opencv_tum.txt"
            if not (args.reuse_existing and expected_output_ready(expected)):
                command = [
                    *command_prefix(cuvslam_python),
                    str(REPO_ROOT / "cuVSLAM" / "examples" / "lerobotv3" / "track_lerobotv3_stereo.py"),
                    "--dataset-root",
                    str(args.dataset_root.expanduser().resolve()),
                    "--episode-index",
                    str(args.episode_index),
                    "--output",
                    str(output_dirs["cuvslam"]),
                    "--left-key",
                    args.left_key,
                    "--right-key",
                    args.right_key,
                    "--camera-params-key",
                    args.camera_params_key,
                    "--gt-column",
                    args.gt_column,
                    "--gt-euler-order",
                    args.gt_euler_order,
                    "--gt-source-frame",
                    args.gt_source_frame,
                    "--gt-pose-convention",
                    args.gt_pose_convention,
                    "--start-frame",
                    str(args.start_frame),
                    "--stride",
                    str(args.stride),
                    "--stereo-t-scale",
                    str(args.stereo_t_scale),
                    "--skip-evo",
                    "--no-rerun",
                ]
                if args.fps is not None:
                    command.extend(["--fps", str(args.fps)])
                if args.end_frame is not None:
                    command.extend(["--end-frame", str(args.end_frame)])
                if args.max_frames is not None:
                    command.extend(["--max-frames", str(args.max_frames)])
                command.extend(shlex.split(args.cuvslam_extra_args))
                run_command(command, cwd=REPO_ROOT, dry_run=args.dry_run, continue_on_error=args.continue_on_error)

        if not args.skip_macvo:
            macvo_python = resolve_python_command(
                args.macvo_python,
                label="macvo",
                env_var="LEROBOT_MACVO_PYTHON",
                env_names=["macvo", "mac-vo"],
                import_check="import torch; import pypose",
                cwd=REPO_ROOT / "MAC-VO",
            )
            expected = output_dirs["macvo"] / "macvo_relative_opencv_tum.txt"
            if not (args.reuse_existing and expected_output_ready(expected)):
                command = [
                    *command_prefix(macvo_python),
                    str(REPO_ROOT / "MAC-VO" / "lerobot_stereo_macvo.py"),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dirs["macvo"]),
                ]
                command.extend(shlex.split(args.macvo_extra_args))
                run_command(command, cwd=REPO_ROOT, dry_run=args.dry_run, continue_on_error=args.continue_on_error)

        if not args.skip_megasam:
            megasam_python = resolve_python_command(
                args.megasam_python,
                label="megasam",
                env_var="LEROBOT_MEGASAM_PYTHON",
                env_names=["mega_sam", "megasam", "mega-sam"],
                import_check="import torch; import cv2",
                cwd=REPO_ROOT / "mega-sam",
            )
            expected = output_dirs["megasam"] / "megasam_relative_opencv_tum.txt"
            if not (args.reuse_existing and expected_output_ready(expected)):
                command = [
                    *command_prefix(megasam_python),
                    str(REPO_ROOT / "mega-sam" / "lerobot_stereo_megasam.py"),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dirs["megasam"]),
                ]
                command.extend(shlex.split(args.megasam_extra_args))
                run_command(command, cwd=REPO_ROOT, dry_run=args.dry_run, continue_on_error=args.continue_on_error)

        if not args.skip_orbslam:
            expected = output_dirs["orbslam"] / "orbslam3_relative_opencv_tum.txt"
            if not (args.reuse_existing and expected_output_ready(expected)):
                command = [
                    sys.executable,
                    str(REPO_ROOT / "ORB_SLAM3" / "lerobot_stereo_orbslam.py"),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dirs["orbslam"]),
                    "--orbslam-root",
                    str(args.orbslam_root.expanduser().resolve()),
                ]
                command.extend(shlex.split(args.orbslam_extra_args))
                run_command(command, cwd=REPO_ROOT, dry_run=args.dry_run, continue_on_error=args.continue_on_error)

    if args.dry_run:
        print("Dry run complete; final aggregation was not run.")
        return

    raw_paths = {
        "cuvslam": output_dirs["cuvslam"] / "cuvslam_stereo_opencv_tum.txt",
        "macvo": output_dirs["macvo"] / "macvo_relative_opencv_tum.txt",
        "megasam": output_dirs["megasam"] / "megasam_relative_opencv_tum.txt",
        "megasam_mono_unidepth": run_root / "megasam_mono_unidepth" / "megasam_relative_opencv_tum.txt",
        "orbslam": output_dirs["orbslam"] / "orbslam3_relative_opencv_tum.txt",
    }

    missing: list[str] = []
    warnings: list[str] = []
    trajectories: list[Trajectory] = []
    max_time_diff = 0.5 / float(manifest["fps"]) + 1e-6

    megasam_label, megasam_note = megasam_label_and_note(args.megasam_extra_args)
    spec = {
        "cuvslam": ("cuVSLAM Stereo", CUVSLAM_COLOR, "stereo odometry on LeRobot rectified pair"),
        "macvo": ("MAC-VO Stereo", MACVO_COLOR, "stereo VO on extracted rectified left/right images"),
        "megasam": (megasam_label, MEGASAM_COLOR, megasam_note),
        "megasam_mono_unidepth": (
            "Mega-SAM Mono (UniDepth)",
            MEGASAM_MONO_UNIDEPTH_COLOR,
            "LeRobot stereo treated as a monocular left-camera video with UniDepth metric depth",
        ),
        "orbslam": ("ORB-SLAM3 Stereo", ORBSLAM_COLOR, "ORB-SLAM3 stereo tracking on extracted rectified left/right images"),
    }

    for key, (label, color, note) in spec.items():
        if getattr(args, f"skip_{key}", False):
            continue
        path = raw_paths[key]
        if key == "megasam_mono_unidepth" and not path.exists():
            continue
        if (
            key == "cuvslam"
            and not path.exists()
            and manifest["episode_index"] == DEFAULT_EPISODE_INDEX
            and DEFAULT_PREBUILT_CUVSLAM_TUM.exists()
        ):
            path = DEFAULT_PREBUILT_CUVSLAM_TUM
        if not path.exists():
            missing.append(f"{key}: missing {path}")
            continue
        rows = load_tum(path)
        trajectories.append(Trajectory(key=key, label=label, rows=rows, color=color, source_path=path, note=note))

    if not trajectories:
        raise FileNotFoundError("No trajectory outputs were found for final comparison.")

    raw_stats = {trajectory.key: trajectory_stats(gt_rows, trajectory.rows, max_time_diff) for trajectory in trajectories}
    display_input_trajectories, tail_trim, trim_warnings = trim_megasam_tail_for_display(
        output_dir,
        trajectories,
        enabled=args.megasam_auto_trim_tail,
        check_frames=args.megasam_tail_check_frames,
        static_step_threshold=args.megasam_tail_static_step,
        min_static_steps=args.megasam_tail_min_static_steps,
    )
    warnings.extend(trim_warnings)

    if args.scale_align:
        if args.alignment_mode == "sim3":
            display_trajectories, scale_alignment, scale_warnings = build_sim3_aligned_trajectories(
                output_dir,
                gt_rows,
                display_input_trajectories,
                gt_tum_path=gt_tum_path,
                reference_key=args.alignment_reference_key,
                max_time_diff=max_time_diff,
                min_displacement=args.scale_align_min_displacement,
                skip_keys=parse_key_list(args.scale_align_skip_keys),
            )
        else:
            display_trajectories, scale_alignment, scale_warnings = build_scale_aligned_trajectories(
                output_dir,
                gt_rows,
                display_input_trajectories,
                reference_path=gt_tum_path,
                max_time_diff=max_time_diff,
                method=args.scale_align_method,
                min_displacement=args.scale_align_min_displacement,
                skip_keys=parse_key_list(args.scale_align_skip_keys),
            )
            scale_alignment["mode"] = "scale"
            scale_alignment["reference_key"] = "gt"
        warnings.extend(scale_warnings)
    else:
        display_trajectories = display_input_trajectories
        scale_alignment = {"enabled": False}

    display_stats = {
        trajectory.key: trajectory_stats(gt_rows, trajectory.rows, max_time_diff)
        for trajectory in display_trajectories
    }
    for key, result in scale_alignment.get("trajectories", {}).items():
        result["stats_vs_gt_after_alignment"] = display_stats.get(key)

    report_path = write_report(
        output_dir,
        manifest=manifest,
        trajectories=trajectories,
        raw_stats=raw_stats,
        display_trajectories=display_trajectories,
        display_stats=display_stats,
        tail_trim=tail_trim,
        scale_alignment=scale_alignment,
        missing=missing,
        warnings=warnings,
    )

    print(f"Saved GT OpenCV TUM: {gt_tum_path}")
    for trajectory in trajectories:
        raw_values = raw_stats.get(trajectory.key)
        display_values = display_stats.get(trajectory.key)
        trim = tail_trim.get("trajectories", {}).get(trajectory.key, {})
        alignment = scale_alignment.get("trajectories", {}).get(trajectory.key, {})
        if raw_values is None:
            print(f"{trajectory.label}: no timestamp matches with GT")
        elif alignment.get("aligned_tum"):
            trim_text = f" | trimmed tail {trim['dropped_frames']} frames" if trim.get("trimmed_tum") else ""
            mode_text = str(scale_alignment.get("mode", "scale"))
            if mode_text == "sim3" and alignment.get("residual_rmse_m") is not None:
                reference_label = scale_alignment.get("reference_label", scale_alignment.get("reference_key", "reference"))
                print(
                    f"{trajectory.label}: {raw_values['frames']} poses | "
                    f"sim3 to {reference_label} scale {float(alignment['scale']):.6g}{trim_text} | "
                    f"reference residual RMSE {float(alignment['residual_rmse_m']):.6f} m "
                    f"(mean {float(alignment['residual_mean_m']):.6f} m, max {float(alignment['residual_max_m']):.6f} m)"
                )
                continue
            if display_values is None:
                print(
                    f"{trajectory.label}: {raw_values['frames']} poses | "
                    f"{mode_text} scale {float(alignment['scale']):.6g}{trim_text} | no aligned stats"
                )
            else:
                print(
                    f"{trajectory.label}: {raw_values['frames']} poses | "
                    f"{mode_text} scale {float(alignment['scale']):.6g}{trim_text} | "
                    f"raw translation RMSE {raw_values['translation_rmse']:.6f} m -> "
                    f"aligned-display {display_values['translation_rmse']:.6f} m | "
                    f"rotation RMSE {display_values['rotation_rmse']:.3f} deg"
                )
        elif alignment.get("skipped"):
            reason = alignment.get("reason", "alignment skipped")
            trim_text = f" | trimmed tail {trim['dropped_frames']} frames" if trim.get("trimmed_tum") else ""
            print(f"{trajectory.label}: {raw_values['frames']} poses{trim_text} | {reason}")
        elif display_values is not None:
            print(
                f"{trajectory.label}: {raw_values['frames']} poses | "
                f"translation RMSE {raw_values['translation_rmse']:.6f} m | "
                f"rotation RMSE {raw_values['rotation_rmse']:.3f} deg"
            )
        else:
            print(
                f"{trajectory.label}: {raw_values['frames']} poses | "
                f"translation RMSE {raw_values['translation_rmse']:.6f} m | "
                f"rotation RMSE {raw_values['rotation_rmse']:.3f} deg | "
                "no display stats"
            )
    if missing:
        print("Missing trajectories:")
        for item in missing:
            print(f"  - {item}")
    if warnings:
        print("Warnings:")
        for item in warnings:
            print(f"  - {item}")
    print(f"Saved report: {report_path}")

    visualize_in_rerun(manifest, gt_rows, display_trajectories, args, output_dir)


if __name__ == "__main__":
    main()
