#!/usr/bin/env python3
"""Run Mega-SAM mono tracking on LeRobot stereo data via a manifest."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MEGASAM_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MEGASAM_ROOT.parent
sys.path.append(str(REPO_ROOT))

from lerobot_v3_common import (  # noqa: E402
    extract_frames_to_directory,
    load_manifest,
    save_tum,
    trajectory_stats,
    world_mats_to_relative_tum,
    write_intrinsics_txt,
)


DEFAULT_WEIGHTS = MEGASAM_ROOT / "checkpoints" / "megasam_final.pth"
DEFAULT_DEPTH_ANYTHING_CKPT = (
    MEGASAM_ROOT / "Depth-Anything" / "checkpoints" / "depth_anything_vitl14.pth"
)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool,
) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def count_files(directory: Path, suffix: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(f"*{suffix}")))


def make_stereo_matcher(num_disparities: int, block_size: int) -> cv2.StereoSGBM:
    channels = 1
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * channels * block_size * block_size,
        P2=32 * channels * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def prepare_stereo_metric_depth(
    left_dir: Path,
    right_dir: Path,
    output_dir: Path,
    *,
    intrinsic: np.ndarray,
    baseline_m: float,
    fov_degrees: float,
    num_disparities: int,
    block_size: int,
    depth_min: float,
    depth_max: float,
) -> int:
    matcher = make_stereo_matcher(num_disparities=num_disparities, block_size=block_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.glob("*.npz"):
        old_path.unlink()

    left_images = sorted(left_dir.glob("*.png"))
    right_images = sorted(right_dir.glob("*.png"))
    if len(left_images) != len(right_images):
        raise ValueError(f"Left/right image count mismatch: {len(left_images)} vs {len(right_images)}")

    fx = float(intrinsic[0, 0])
    written = 0
    for index, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        left_gray = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right_gray = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left_gray is None or right_gray is None:
            raise FileNotFoundError(f"Failed to read stereo pair: {left_path}, {right_path}")

        disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        valid = np.isfinite(disparity) & (disparity > 0.5)

        depth = np.full(disparity.shape, float(depth_max), dtype=np.float32)
        depth[valid] = (fx * float(baseline_m)) / disparity[valid]
        depth = np.clip(depth, float(depth_min), float(depth_max))

        np.savez(
            output_dir / f"{index:06d}.npz",
            depth=depth.astype(np.float32),
            fov=np.asarray(float(fov_degrees), dtype=np.float32),
            stereo_valid=valid.astype(np.uint8),
        )
        written += 1
    return written


def run_depth_anything(
    *,
    images_dir: Path,
    mono_root: Path,
    scene_name: str,
    checkpoint: Path,
    encoder: str,
    dry_run: bool,
) -> None:
    scene_dir = mono_root / scene_name
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    run_command(
        [
            sys.executable,
            "Depth-Anything/run_videos.py",
            "--encoder",
            encoder,
            "--load-from",
            str(checkpoint),
            "--img-path",
            str(images_dir),
            "--outdir",
            str(scene_dir),
        ],
        cwd=MEGASAM_ROOT,
        dry_run=dry_run,
    )


def run_unidepth_metric_depth(
    *,
    images_dir: Path,
    metric_root: Path,
    scene_name: str,
    dry_run: bool,
) -> None:
    scene_dir = metric_root / scene_name
    if scene_dir.exists():
        shutil.rmtree(scene_dir)

    env = os.environ.copy()
    unidepth_path = str(MEGASAM_ROOT / "UniDepth")
    env["PYTHONPATH"] = (
        unidepth_path
        if not env.get("PYTHONPATH")
        else f"{unidepth_path}:{env['PYTHONPATH']}"
    )
    run_command(
        [
            sys.executable,
            "UniDepth/scripts/demo_mega-sam.py",
            "--scene-name",
            scene_name,
            "--img-path",
            str(images_dir),
            "--outdir",
            str(metric_root),
        ],
        cwd=MEGASAM_ROOT,
        env=env,
        dry_run=dry_run,
    )


def remove_megasam_scene_outputs(scene_name: str) -> None:
    npz_path = MEGASAM_ROOT / "outputs" / f"{scene_name}_droid.npz"
    if npz_path.exists():
        npz_path.unlink()
    reconstruction_dir = MEGASAM_ROOT / "reconstructions" / scene_name
    if reconstruction_dir.exists():
        shutil.rmtree(reconstruction_dir)


def run_megasam_tracking(
    *,
    images_dir: Path,
    scene_name: str,
    mono_root: Path,
    metric_root: Path,
    depth_source: str,
    intrinsic_path: Path,
    weights: Path,
    buffer: int,
    beta: float,
    filter_thresh: float,
    warmup: int,
    keyframe_thresh: float,
    frontend_thresh: float,
    frontend_window: int,
    frontend_radius: int,
    frontend_nms: int,
    backend_thresh: float,
    backend_radius: int,
    backend_nms: int,
    disable_full_ba: bool,
    upsample: bool,
    opt_focal: bool,
    dry_run: bool,
) -> Path:
    command = [
        sys.executable,
        "camera_tracking_scripts/test_demo.py",
        "--datapath",
        str(images_dir),
        "--weights",
        str(weights),
        "--scene_name",
        scene_name,
        "--mono_depth_path",
        str(mono_root),
        "--metric_depth_path",
        str(metric_root),
        "--intrinsics",
        str(intrinsic_path),
        "--depth_source",
        depth_source,
        "--disable_vis",
        "--buffer",
        str(buffer),
        "--beta",
        str(beta),
        "--filter_thresh",
        str(filter_thresh),
        "--warmup",
        str(warmup),
        "--keyframe_thresh",
        str(keyframe_thresh),
        "--frontend_thresh",
        str(frontend_thresh),
        "--frontend_window",
        str(frontend_window),
        "--frontend_radius",
        str(frontend_radius),
        "--frontend_nms",
        str(frontend_nms),
        "--backend_thresh",
        str(backend_thresh),
        "--backend_radius",
        str(backend_radius),
        "--backend_nms",
        str(backend_nms),
    ]
    if disable_full_ba:
        command.append("--disable_full_ba")
    if upsample:
        command.append("--upsample")
    if opt_focal:
        command.append("--opt_focal")

    run_command(command, cwd=MEGASAM_ROOT, dry_run=dry_run)
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


def write_report(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    stats: dict[str, float | int] | None,
    paths: dict[str, str],
    metric_depth_source: str,
    depth_source: str,
) -> Path:
    if metric_depth_source == "stereo":
        scale_policy = (
            "Mega-SAM tracks the left camera; metric depth is computed from the rectified "
            "right camera by stereo SGBM."
        )
    else:
        scale_policy = (
            "Mega-SAM treats the LeRobot stereo episode as a monocular left-camera video; "
            "metric depth is predicted from the left images by UniDepth."
        )
    report = {
        "scene_name": manifest["scene_name"],
        "dataset_root": manifest["dataset_root"],
        "episode_index": manifest["episode_index"],
        "fps": manifest["fps"],
        "coordinate_note": manifest["coordinate_note"],
        "scale_policy": scale_policy,
        "metric_depth_source": metric_depth_source,
        "depth_source": depth_source,
        "trajectory_policy": "Full BA is disabled in this wrapper so Mega-SAM returns per-frame poses compatible with manifest timestamps.",
        "stats_vs_gt": stats,
        "paths": paths,
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    text_path = output_dir / "comparison_report.txt"
    with text_path.open("w", encoding="utf-8") as file:
        file.write("LeRobot v3 stereo Mega-SAM report\n")
        file.write(f"scene_name={manifest['scene_name']}\n")
        file.write(f"episode_index={manifest['episode_index']}\n")
        file.write(f"dataset_root={manifest['dataset_root']}\n")
        file.write(f"fps={manifest['fps']:.6f}\n")
        file.write("coordinate=first-selected-frame-relative OpenCV optical camera (x right, y down, z forward)\n")
        file.write(f"metric_depth_source={metric_depth_source}\n")
        file.write(f"depth_source={depth_source}\n")
        file.write(f"scale_policy={scale_policy}\n\n")
        file.write("trajectory_policy=full BA disabled for per-frame timestamp alignment\n\n")
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
        description="Run Mega-SAM mono tracking on LeRobot stereo data described by a manifest JSON."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--depth-anything-checkpoint", type=Path, default=DEFAULT_DEPTH_ANYTHING_CKPT)
    parser.add_argument("--depth-anything-encoder", choices=("vits", "vitb", "vitl"), default="vitl")
    parser.add_argument("--skip-extract", action="store_true", help="reuse extracted png frames")
    parser.add_argument("--skip-depth", action="store_true", help="reuse existing depth outputs")
    parser.add_argument("--skip-tracking", action="store_true", help="reuse existing Mega-SAM npz")
    parser.add_argument(
        "--depth-source",
        choices=("metric", "mono_aligned"),
        default="metric",
        help="metric feeds metric depth directly; mono_aligned aligns Depth-Anything to the metric-depth source",
    )
    parser.add_argument(
        "--metric-depth-source",
        choices=("stereo", "unidepth"),
        default="stereo",
        help="stereo uses the rectified right camera; unidepth treats the stereo episode as monocular left-camera video",
    )
    parser.add_argument("--keep-megasam-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.3)
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
    parser.add_argument(
        "--disable-full-ba",
        action="store_true",
        help="accepted for compatibility; this wrapper always disables full BA to keep per-frame trajectories",
    )
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--opt-focal", action="store_true")
    parser.add_argument("--stereo-num-disparities", type=int, default=128, help="must be a multiple of 16")
    parser.add_argument("--stereo-block-size", type=int, default=5)
    parser.add_argument("--depth-min", type=float, default=0.15)
    parser.add_argument("--depth-max", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stereo_num_disparities < 16 or args.stereo_num_disparities % 16 != 0:
        raise ValueError("--stereo-num-disparities must be a positive multiple of 16")
    if args.stereo_block_size < 3 or args.stereo_block_size % 2 == 0:
        raise ValueError("--stereo-block-size must be an odd integer >= 3")

    manifest = load_manifest(args.manifest)
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = np.asarray(manifest["timestamps"], dtype=np.float64)
    gt_rows = np.asarray(manifest["gt_rows"], dtype=np.float64)
    calibration = manifest["calibration"]
    intrinsic = np.asarray(calibration["rectified_left_K"], dtype=np.float64)
    baseline_m = float(calibration["baseline_m"])
    fov_degrees = float(calibration["fov_degrees"])
    scene_name = manifest["scene_name"]

    images_dir = output_dir / "images"
    mono_root = output_dir / "Depth-Anything" / "video_visualization"
    right_dir = output_dir / "right_images"
    metric_root = (
        output_dir / "StereoDepth" / "outputs"
        if args.metric_depth_source == "stereo"
        else output_dir / "UniDepth" / "outputs"
    )
    metric_scene_dir = metric_root / scene_name

    intrinsics_path = write_intrinsics_txt(output_dir / "camera_intrinsic.txt", intrinsic)
    gt_tum_path = save_tum(output_dir / "gt_relative_opencv_tum.txt", gt_rows)
    np.savetxt(output_dir / "frame_ids.txt", np.asarray(manifest["frame_ids"], dtype=np.int64), fmt="%d")
    np.savetxt(output_dir / "timestamps.txt", timestamps, fmt="%.6f")

    if args.skip_extract:
        if not images_dir.exists():
            raise FileNotFoundError("Missing extracted left image directory for --skip-extract")
        if args.metric_depth_source == "stereo" and not right_dir.exists():
            raise FileNotFoundError("Missing extracted right image directory for --skip-extract with stereo depth")
    else:
        extract_frames_to_directory(
            Path(manifest["left_video"]["path"]),
            manifest["left_video_frame_ids"],
            images_dir,
            fps=float(manifest["fps"]),
        )
        if args.metric_depth_source == "stereo":
            extract_frames_to_directory(
                Path(manifest["right_video"]["path"]),
                manifest["right_video_frame_ids"],
                right_dir,
                fps=float(manifest["fps"]),
            )

    if not args.skip_depth:
        if args.metric_depth_source == "stereo":
            stereo_count = prepare_stereo_metric_depth(
                images_dir,
                right_dir,
                metric_scene_dir,
                intrinsic=intrinsic,
                baseline_m=baseline_m,
                fov_degrees=fov_degrees,
                num_disparities=args.stereo_num_disparities,
                block_size=args.stereo_block_size,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
            )
            print(f"Prepared stereo metric depth: {stereo_count} frames")
        else:
            run_unidepth_metric_depth(
                images_dir=images_dir,
                metric_root=metric_root,
                scene_name=scene_name,
                dry_run=args.dry_run,
            )
        if args.depth_source == "mono_aligned":
            args.depth_anything_checkpoint = args.depth_anything_checkpoint.expanduser().resolve()
            if not args.depth_anything_checkpoint.exists():
                raise FileNotFoundError(f"Missing Depth-Anything checkpoint: {args.depth_anything_checkpoint}")
            run_depth_anything(
                images_dir=images_dir,
                mono_root=mono_root,
                scene_name=scene_name,
                checkpoint=args.depth_anything_checkpoint,
                encoder=args.depth_anything_encoder,
                dry_run=args.dry_run,
            )
    elif not metric_scene_dir.exists():
        raise FileNotFoundError(f"Missing stereo depth directory for --skip-depth: {metric_scene_dir}")
    elif args.depth_source == "mono_aligned" and not (mono_root / scene_name).exists():
        raise FileNotFoundError(f"Missing mono depth directory for --skip-depth: {mono_root / scene_name}")

    if args.dry_run:
        if not args.skip_tracking:
            run_megasam_tracking(
                images_dir=images_dir,
                scene_name=scene_name,
                mono_root=mono_root,
                metric_root=metric_root,
                depth_source=args.depth_source,
                intrinsic_path=intrinsics_path,
                weights=args.weights.expanduser().resolve(),
                buffer=args.buffer,
                beta=args.beta,
                filter_thresh=args.filter_thresh,
                warmup=args.warmup,
                keyframe_thresh=args.keyframe_thresh,
                frontend_thresh=args.frontend_thresh,
                frontend_window=args.frontend_window,
                frontend_radius=args.frontend_radius,
                frontend_nms=args.frontend_nms,
                backend_thresh=args.backend_thresh,
                backend_radius=args.backend_radius,
                backend_nms=args.backend_nms,
                disable_full_ba=True,
                upsample=args.upsample,
                opt_focal=args.opt_focal,
                dry_run=True,
            )
        print("dry-run complete; Mega-SAM output conversion was not run.")
        return

    mono_count = count_files(mono_root / scene_name, ".npy")
    metric_count = count_files(metric_scene_dir, ".npz")
    if metric_count < len(timestamps) or (args.depth_source == "mono_aligned" and mono_count < len(timestamps)):
        raise RuntimeError(
            f"Incomplete depth outputs: mono {mono_count}/{len(timestamps)}, metric {metric_count}/{len(timestamps)}"
        )

    if args.skip_tracking:
        megasam_npz_path = output_dir / "megasam_droid.npz"
        if not megasam_npz_path.exists():
            fallback = MEGASAM_ROOT / "outputs" / f"{scene_name}_droid.npz"
            if not fallback.exists():
                raise FileNotFoundError("Missing Mega-SAM npz for --skip-tracking")
            megasam_npz_path = fallback
    else:
        args.weights = args.weights.expanduser().resolve()
        if not args.weights.exists():
            raise FileNotFoundError(f"Missing Mega-SAM checkpoint: {args.weights}")
        if not args.keep_megasam_cache:
            remove_megasam_scene_outputs(scene_name)
        tracking_buffer = max(int(args.buffer), int(len(timestamps)))
        megasam_npz_path = run_megasam_tracking(
            images_dir=images_dir,
            scene_name=scene_name,
            mono_root=mono_root,
            metric_root=metric_root,
            depth_source=args.depth_source,
            intrinsic_path=intrinsics_path,
            weights=args.weights,
            buffer=tracking_buffer,
            beta=args.beta,
            filter_thresh=args.filter_thresh,
            warmup=args.warmup,
            keyframe_thresh=args.keyframe_thresh,
            frontend_thresh=args.frontend_thresh,
            frontend_window=args.frontend_window,
            frontend_radius=args.frontend_radius,
            frontend_nms=args.frontend_nms,
            backend_thresh=args.backend_thresh,
            backend_radius=args.backend_radius,
            backend_nms=args.backend_nms,
            disable_full_ba=True,
            upsample=args.upsample,
            opt_focal=args.opt_focal,
            dry_run=False,
        )
        if not megasam_npz_path.exists():
            raise FileNotFoundError(f"Mega-SAM did not create expected output: {megasam_npz_path}")
        shutil.copy2(megasam_npz_path, output_dir / "megasam_droid.npz")
        megasam_npz_path = output_dir / "megasam_droid.npz"

    cam_c2w = load_megasam_cam_c2w(megasam_npz_path)
    count = min(len(cam_c2w), len(timestamps), len(gt_rows))
    cam_c2w = cam_c2w[:count]
    timestamps = timestamps[:count]
    gt_rows = gt_rows[:count]
    gt_tum_path = save_tum(output_dir / "gt_relative_opencv_tum.txt", gt_rows)

    megasam_rows = world_mats_to_relative_tum(cam_c2w, timestamps)
    megasam_tum_path = save_tum(output_dir / "megasam_relative_opencv_tum.txt", megasam_rows)
    stats = trajectory_stats(gt_rows, megasam_rows, max_time_diff=0.5 / float(manifest["fps"]) + 1e-6)

    paths = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "frame_ids": str(output_dir / "frame_ids.txt"),
        "timestamps": str(output_dir / "timestamps.txt"),
        "gt_tum": str(gt_tum_path),
        "intrinsics": str(intrinsics_path),
        "images": str(images_dir),
        "right_images": str(right_dir) if args.metric_depth_source == "stereo" else "",
        "depth_source": args.depth_source,
        "metric_depth_source": args.metric_depth_source,
        "mono_depth": str(mono_root / scene_name) if args.depth_source == "mono_aligned" else "",
        "metric_depth": str(metric_scene_dir),
        "megasam_npz": str(megasam_npz_path),
        "megasam_tum": str(megasam_tum_path),
    }
    report_path = write_report(
        output_dir,
        manifest=manifest,
        stats=stats,
        paths=paths,
        metric_depth_source=args.metric_depth_source,
        depth_source=args.depth_source,
    )

    print(f"Output directory: {output_dir}")
    print(f"Saved GT OpenCV TUM: {gt_tum_path}")
    print(f"Saved Mega-SAM OpenCV TUM: {megasam_tum_path}")
    print(f"Saved report: {report_path}")
    if stats is not None:
        print(
            "Mega-SAM | "
            f"translation RMSE {stats['translation_rmse']:.6f} m | "
            f"rotation RMSE {stats['rotation_rmse']:.3f} deg"
        )


if __name__ == "__main__":
    main()
