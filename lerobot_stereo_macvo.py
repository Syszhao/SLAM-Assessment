#!/usr/bin/env python3
"""Run MAC-VO stereo on a prepared LeRobot v3 manifest."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


MACVO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MACVO_ROOT.parent
sys.path.append(str(REPO_ROOT))

from lerobot_v3_common import (  # noqa: E402
    OPENCV_FROM_MACVO_NED_CAMERA,
    extract_frames_to_directory,
    load_manifest,
    save_tum,
    se3_row_to_matrix,
    trajectory_stats,
    world_mats_to_relative_tum,
    write_intrinsics_txt,
)


DEFAULT_ODOM_CONFIG = MACVO_ROOT / "Config" / "Experiment" / "MACVO" / "MACVO_Performant.yaml"


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def write_data_config(path: Path, *, dataset_root: Path, intrinsic: np.ndarray, baseline_m: float, scene_name: str) -> Path:
    payload = {
        "type": "GeneralStereo",
        "name": scene_name,
        "args": {
            "root": str(dataset_root),
            "camera": {
                "fx": float(intrinsic[0, 0]),
                "fy": float(intrinsic[1, 1]),
                "cx": float(intrinsic[0, 2]),
                "cy": float(intrinsic[1, 2]),
            },
            "bl": float(baseline_m),
            "format": "png",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def find_latest_pose_file(results_root: Path) -> Path:
    candidates = sorted(results_root.glob("**/poses.npy"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No MAC-VO poses.npy found under {results_root}")
    return candidates[-1]


def load_macvo_rows(poses_path: Path, timestamps: np.ndarray) -> np.ndarray:
    pose_data = np.load(poses_path).astype(np.float64)
    if pose_data.ndim != 2 or pose_data.shape[1] < 8:
        raise ValueError(f"MAC-VO poses.npy must have shape Nx8+, got {pose_data.shape}")

    count = min(len(pose_data), len(timestamps))
    if count == 0:
        raise ValueError("MAC-VO returned zero poses")

    pose_mats = np.stack([se3_row_to_matrix(row[1:8]) for row in pose_data[:count]])
    return world_mats_to_relative_tum(
        pose_mats,
        timestamps[:count],
        target_from_source=OPENCV_FROM_MACVO_NED_CAMERA,
    )


def write_report(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    pose_file: Path,
    sandbox_dir: Path,
    stats: dict[str, float | int] | None,
    paths: dict[str, str],
) -> Path:
    report = {
        "scene_name": manifest["scene_name"],
        "dataset_root": manifest["dataset_root"],
        "episode_index": manifest["episode_index"],
        "fps": manifest["fps"],
        "coordinate_note": manifest["coordinate_note"],
        "macvo_source_pose_file": str(pose_file),
        "macvo_sandbox_dir": str(sandbox_dir),
        "stats_vs_gt": stats,
        "paths": paths,
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    text_path = output_dir / "comparison_report.txt"
    with text_path.open("w", encoding="utf-8") as file:
        file.write("LeRobot v3 stereo MAC-VO report\n")
        file.write(f"scene_name={manifest['scene_name']}\n")
        file.write(f"episode_index={manifest['episode_index']}\n")
        file.write(f"dataset_root={manifest['dataset_root']}\n")
        file.write(f"fps={manifest['fps']:.6f}\n")
        file.write(f"macvo_source_pose_file={pose_file}\n")
        file.write(f"macvo_sandbox_dir={sandbox_dir}\n")
        file.write("coordinate=first-selected-frame-relative OpenCV optical camera (x right, y down, z forward)\n\n")
        if stats is None:
            file.write("stats=None\n")
        else:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MAC-VO stereo on LeRobot v3 frames described by a manifest JSON."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--odom-config", type=Path, default=DEFAULT_ODOM_CONFIG)
    parser.add_argument("--skip-extract", action="store_true", help="reuse left/right pngs if present")
    parser.add_argument("--skip-run", action="store_true", help="reuse latest poses.npy under --result-root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--result-root", type=Path, help="MAC-VO result root; default is <output>/macvo_results")
    parser.add_argument("--extra-args", default="", help="extra args appended to MACVO.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = np.asarray(manifest["timestamps"], dtype=np.float64)
    gt_rows = np.asarray(manifest["gt_rows"], dtype=np.float64)
    calibration = manifest["calibration"]
    intrinsic = np.asarray(calibration["rectified_left_K"], dtype=np.float64)
    baseline_m = float(calibration["baseline_m"])

    dataset_dir = output_dir / "dataset"
    left_dir = dataset_dir / "left"
    right_dir = dataset_dir / "right"
    data_config = write_data_config(
        output_dir / "config" / "sequence.yaml",
        dataset_root=dataset_dir,
        intrinsic=intrinsic,
        baseline_m=baseline_m,
        scene_name=manifest["scene_name"],
    )
    intrinsics_path = write_intrinsics_txt(output_dir / "camera_intrinsic.txt", intrinsic)
    gt_tum_path = save_tum(output_dir / "gt_relative_opencv_tum.txt", gt_rows)
    np.savetxt(output_dir / "frame_ids.txt", np.asarray(manifest["frame_ids"], dtype=np.int64), fmt="%d")
    np.savetxt(output_dir / "timestamps.txt", timestamps, fmt="%.6f")

    if args.skip_extract:
        if not left_dir.exists() or not right_dir.exists():
            raise FileNotFoundError("Missing extracted left/right image directories for --skip-extract")
    else:
        extract_frames_to_directory(
            Path(manifest["left_video"]["path"]),
            manifest["left_video_frame_ids"],
            left_dir,
            fps=float(manifest["fps"]),
        )
        extract_frames_to_directory(
            Path(manifest["right_video"]["path"]),
            manifest["right_video_frame_ids"],
            right_dir,
            fps=float(manifest["fps"]),
        )

    result_root = (args.result_root or (output_dir / "macvo_results")).expanduser().resolve()
    pose_file: Path
    if args.skip_run:
        pose_file = find_latest_pose_file(result_root)
    else:
        args.odom_config = args.odom_config.expanduser().resolve()
        if not args.odom_config.exists():
            raise FileNotFoundError(f"Missing odom config: {args.odom_config}")
        command = [
            sys.executable,
            "MACVO.py",
            "--odom",
            str(args.odom_config),
            "--data",
            str(data_config),
            "--resultRoot",
            str(result_root),
            "--noeval",
        ]
        command.extend(shlex.split(args.extra_args))
        run_command(command, cwd=MACVO_ROOT, dry_run=args.dry_run)
        if args.dry_run:
            print("dry-run complete; MAC-VO output conversion was not run.")
            return
        pose_file = find_latest_pose_file(result_root)

    sandbox_dir = pose_file.parent
    macvo_rows = load_macvo_rows(pose_file, timestamps)
    count = min(len(macvo_rows), len(gt_rows))
    macvo_rows = macvo_rows[:count]
    gt_rows = gt_rows[:count]
    gt_tum_path = save_tum(output_dir / "gt_relative_opencv_tum.txt", gt_rows)
    macvo_tum_path = save_tum(output_dir / "macvo_relative_opencv_tum.txt", macvo_rows)

    stats = trajectory_stats(gt_rows, macvo_rows, max_time_diff=0.5 / float(manifest["fps"]) + 1e-6)
    paths = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "frame_ids": str(output_dir / "frame_ids.txt"),
        "timestamps": str(output_dir / "timestamps.txt"),
        "gt_tum": str(gt_tum_path),
        "intrinsics": str(intrinsics_path),
        "data_config": str(data_config),
        "macvo_tum": str(macvo_tum_path),
        "macvo_pose_file": str(pose_file),
    }
    report_path = write_report(
        output_dir,
        manifest=manifest,
        pose_file=pose_file,
        sandbox_dir=sandbox_dir,
        stats=stats,
        paths=paths,
    )

    print(f"Output directory: {output_dir}")
    print(f"MAC-VO pose source: {pose_file}")
    print(f"Saved GT OpenCV TUM: {gt_tum_path}")
    print(f"Saved MAC-VO OpenCV TUM: {macvo_tum_path}")
    print(f"Saved report: {report_path}")
    if stats is not None:
        print(
            "MAC-VO | "
            f"translation RMSE {stats['translation_rmse']:.6f} m | "
            f"rotation RMSE {stats['rotation_rmse']:.3f} deg"
        )


if __name__ == "__main__":
    main()
