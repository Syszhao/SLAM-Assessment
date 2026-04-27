#!/usr/bin/env python3
"""Run ORB-SLAM3 stereo on a prepared LeRobot v3 manifest."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ORBSLAM_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ORBSLAM_ROOT.parent
sys.path.append(str(REPO_ROOT))

from lerobot_v3_common import (  # noqa: E402
    iter_video_frames,
    load_manifest,
    normalize_quaternions,
    save_tum,
    trajectory_stats,
    tum_row_to_matrix,
    world_mats_to_relative_tum,
)


DEFAULT_TRAJECTORY_NAME = "orbslam3_lerobot"


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def require_file(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def default_executable(root: Path) -> Path:
    lerobot_executable = root / "Examples" / "Stereo" / "stereo_lerobot"
    if lerobot_executable.exists():
        return lerobot_executable
    return root / "Examples" / "Stereo" / "stereo_euroc"


def timestamp_ns_labels(timestamps: np.ndarray) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for timestamp in np.asarray(timestamps, dtype=np.float64):
        label = str(int(round(float(timestamp) * 1_000_000_000.0)))
        if label in seen:
            raise ValueError(f"Duplicate ORB-SLAM3 nanosecond timestamp label: {label}")
        labels.append(label)
        seen.add(label)
    return labels


def extract_frames_with_names(
    video_path: Path,
    frame_ids: list[int],
    labels: list[str],
    output_dir: Path,
    *,
    fps: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.glob("*.png"):
        old_path.unlink()

    written: list[Path] = []
    for index, (_, image_rgb) in enumerate(iter_video_frames(video_path, frame_ids, fps)):
        output_path = output_dir / f"{labels[index]}.png"
        frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_path), frame_bgr):
            raise IOError(f"Failed to write frame: {output_path}")
        written.append(output_path)

    if len(written) != len(labels):
        raise RuntimeError(f"Decoded {len(written)}/{len(labels)} requested frames from {video_path}")
    return written


def write_times_file(path: Path, labels: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return path


def write_orbslam_settings(
    path: Path,
    *,
    calibration: dict[str, Any],
    fps: float,
    n_features: int,
    scale_factor: float,
    n_levels: int,
    ini_th_fast: int,
    min_th_fast: int,
    stereo_th_depth: float,
) -> Path:
    image_width, image_height = [int(value) for value in calibration["image_size"]]
    intrinsic = np.asarray(calibration["rectified_left_K"], dtype=np.float64)
    baseline_m = float(calibration["baseline_m"])
    fps_int = max(1, int(round(float(fps))))

    def f(value: float) -> str:
        return f"{float(value):.9f}"

    text = f"""%YAML:1.0

File.version: "1.0"

Camera.type: "Rectified"

Camera1.fx: {f(intrinsic[0, 0])}
Camera1.fy: {f(intrinsic[1, 1])}
Camera1.cx: {f(intrinsic[0, 2])}
Camera1.cy: {f(intrinsic[1, 2])}

Camera.width: {image_width}
Camera.height: {image_height}
Camera.fps: {fps_int}
Camera.RGB: 0

Stereo.b: {f(baseline_m)}
Stereo.ThDepth: {f(stereo_th_depth)}

ORBextractor.nFeatures: {int(n_features)}
ORBextractor.scaleFactor: {f(scale_factor)}
ORBextractor.nLevels: {int(n_levels)}
ORBextractor.iniThFAST: {int(ini_th_fast)}
ORBextractor.minThFAST: {int(min_th_fast)}

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def load_orbslam_euroc_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise ValueError(f"ORB-SLAM3 EuRoC trajectory must have shape Nx8: {path}")

    finite = np.isfinite(rows[:, 0:8]).all(axis=1)
    rows = rows[finite]
    if len(rows) == 0:
        raise ValueError(f"ORB-SLAM3 trajectory has no finite poses: {path}")

    raw_seconds = rows.copy()
    raw_seconds[:, 0] = raw_seconds[:, 0] / 1_000_000_000.0
    raw_seconds[:, 4:8] = normalize_quaternions(raw_seconds[:, 4:8])

    world_from_camera = np.stack([tum_row_to_matrix(row) for row in raw_seconds])
    relative_rows = world_mats_to_relative_tum(world_from_camera, raw_seconds[:, 0])
    return raw_seconds, relative_rows


def write_report(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    stats: dict[str, float | int] | None,
    paths: dict[str, str],
) -> Path:
    report = {
        "scene_name": manifest["scene_name"],
        "dataset_root": manifest["dataset_root"],
        "episode_index": manifest["episode_index"],
        "fps": manifest["fps"],
        "coordinate_note": manifest["coordinate_note"],
        "orbslam_policy": (
            "ORB-SLAM3 stereo is run on rectified LeRobot left/right frames prepared in EuRoC layout; "
            "EuRoC nanosecond output is converted to first-localized-frame-relative OpenCV TUM seconds."
        ),
        "stats_vs_gt": stats,
        "paths": paths,
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    text_path = output_dir / "comparison_report.txt"
    with text_path.open("w", encoding="utf-8") as file:
        file.write("LeRobot v3 stereo ORB-SLAM3 report\n")
        file.write(f"scene_name={manifest['scene_name']}\n")
        file.write(f"episode_index={manifest['episode_index']}\n")
        file.write(f"dataset_root={manifest['dataset_root']}\n")
        file.write(f"fps={manifest['fps']:.6f}\n")
        file.write("coordinate=first-localized-frame-relative OpenCV optical camera (x right, y down, z forward)\n\n")
        if stats is None:
            file.write("stats=None\n\n")
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
        description="Run ORB-SLAM3 stereo on LeRobot v3 frames described by a manifest JSON."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orbslam-root", type=Path, default=ORBSLAM_ROOT)
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--settings", type=Path, help="use an existing ORB-SLAM3 YAML instead of generating one")
    parser.add_argument("--trajectory-name", default=DEFAULT_TRAJECTORY_NAME)
    parser.add_argument("--skip-extract", action="store_true", help="reuse prepared EuRoC image folders")
    parser.add_argument("--skip-run", action="store_true", help="reuse ORB-SLAM3 f_<trajectory-name>.txt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--orb-features", type=int, default=2000)
    parser.add_argument("--orb-scale-factor", type=float, default=1.2)
    parser.add_argument("--orb-levels", type=int, default=8)
    parser.add_argument("--orb-ini-th-fast", type=int, default=20)
    parser.add_argument("--orb-min-th-fast", type=int, default=7)
    parser.add_argument("--orb-stereo-th-depth", type=float, default=40.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    orbslam_root = args.orbslam_root.expanduser().resolve()
    vocabulary = (args.vocabulary or (orbslam_root / "Vocabulary" / "ORBvoc.txt")).expanduser().resolve()
    executable = (args.executable or default_executable(orbslam_root)).expanduser().resolve()
    require_file(vocabulary, "ORB-SLAM3 vocabulary")
    require_file(executable, "ORB-SLAM3 stereo executable")

    timestamps = np.asarray(manifest["timestamps"], dtype=np.float64)
    gt_rows = np.asarray(manifest["gt_rows"], dtype=np.float64)
    labels = timestamp_ns_labels(timestamps)

    sequence_dir = output_dir / "euroc_sequence"
    left_dir = sequence_dir / "mav0" / "cam0" / "data"
    right_dir = sequence_dir / "mav0" / "cam1" / "data"
    times_path = write_times_file(output_dir / "euroc_timestamps_ns.txt", labels)
    gt_tum_path = save_tum(output_dir / "gt_relative_opencv_tum.txt", gt_rows)
    np.savetxt(output_dir / "frame_ids.txt", np.asarray(manifest["frame_ids"], dtype=np.int64), fmt="%d")
    np.savetxt(output_dir / "timestamps.txt", timestamps, fmt="%.6f")

    if args.skip_extract:
        if len(list(left_dir.glob("*.png"))) < len(labels) or len(list(right_dir.glob("*.png"))) < len(labels):
            raise FileNotFoundError("Missing prepared EuRoC image folders for --skip-extract")
    else:
        extract_frames_with_names(
            Path(manifest["left_video"]["path"]),
            manifest["left_video_frame_ids"],
            labels,
            left_dir,
            fps=float(manifest["fps"]),
        )
        extract_frames_with_names(
            Path(manifest["right_video"]["path"]),
            manifest["right_video_frame_ids"],
            labels,
            right_dir,
            fps=float(manifest["fps"]),
        )

    if args.settings is None:
        settings_path = write_orbslam_settings(
            output_dir / "config" / "lerobot_orbslam3.yaml",
            calibration=manifest["calibration"],
            fps=float(manifest["fps"]),
            n_features=args.orb_features,
            scale_factor=args.orb_scale_factor,
            n_levels=args.orb_levels,
            ini_th_fast=args.orb_ini_th_fast,
            min_th_fast=args.orb_min_th_fast,
            stereo_th_depth=args.orb_stereo_th_depth,
        )
    else:
        settings_path = args.settings.expanduser().resolve()
        require_file(settings_path, "ORB-SLAM3 settings")

    raw_trajectory_path = output_dir / f"f_{args.trajectory_name}.txt"
    raw_keyframe_path = output_dir / f"kf_{args.trajectory_name}.txt"

    if args.skip_run:
        require_file(raw_trajectory_path, "ORB-SLAM3 raw frame trajectory")
    else:
        command = [
            str(executable),
            str(vocabulary),
            str(settings_path),
            str(sequence_dir),
            str(times_path),
            args.trajectory_name,
        ]
        run_command(command, cwd=output_dir, dry_run=args.dry_run)
        if args.dry_run:
            print("dry-run complete; ORB-SLAM3 output conversion was not run.")
            return
        require_file(raw_trajectory_path, "ORB-SLAM3 raw frame trajectory")

    raw_seconds_rows, orbslam_rows = load_orbslam_euroc_trajectory(raw_trajectory_path)
    raw_seconds_path = save_tum(output_dir / "orbslam3_raw_world_seconds_tum.txt", raw_seconds_rows)
    orbslam_tum_path = save_tum(output_dir / "orbslam3_relative_opencv_tum.txt", orbslam_rows)

    stats = trajectory_stats(gt_rows, orbslam_rows, max_time_diff=0.5 / float(manifest["fps"]) + 1e-6)
    paths = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "frame_ids": str(output_dir / "frame_ids.txt"),
        "timestamps": str(output_dir / "timestamps.txt"),
        "gt_tum": str(gt_tum_path),
        "euroc_sequence": str(sequence_dir),
        "euroc_timestamps_ns": str(times_path),
        "settings": str(settings_path),
        "vocabulary": str(vocabulary),
        "executable": str(executable),
        "orbslam_raw_euroc": str(raw_trajectory_path),
        "orbslam_raw_keyframes": str(raw_keyframe_path),
        "orbslam_raw_seconds_tum": str(raw_seconds_path),
        "orbslam_tum": str(orbslam_tum_path),
    }
    report_path = write_report(output_dir, manifest=manifest, stats=stats, paths=paths)

    print(f"Output directory: {output_dir}")
    print(f"Saved GT OpenCV TUM: {gt_tum_path}")
    print(f"Saved ORB-SLAM3 OpenCV TUM: {orbslam_tum_path}")
    print(f"Saved report: {report_path}")
    if stats is not None:
        print(
            "ORB-SLAM3 | "
            f"translation RMSE {stats['translation_rmse']:.6f} m | "
            f"rotation RMSE {stats['rotation_rmse']:.3f} deg"
        )


if __name__ == "__main__":
    main()
