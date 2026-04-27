import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


DROID_ROOT = Path(__file__).resolve().parent
sys.path.append(str(DROID_ROOT / "droid_slam"))

try:
    import h5py
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少 h5py。请在 DROID 环境中安装：\n"
        "  conda run -n droid-slam python -m pip install h5py"
    ) from exc

try:
    from droid import Droid
    from droid_async import DroidAsync
except ModuleNotFoundError as exc:
    missing_module = exc.name or "unknown"
    raise SystemExit(
        "无法导入 DROID-SLAM 模块。\n"
        f"实际缺少的 Python 模块: {missing_module}\n"
        "请确认使用 droid-slam 环境，并补装缺失依赖，例如：\n"
        f"  conda run -n droid-slam python -m pip install {missing_module}\n"
        "然后从 /data/evo 附近运行：\n"
        "  conda run -n droid-slam python DROID-SLAM/egodex_rerun.py ..."
    ) from exc


DEFAULT_HDF5 = "/home/user/test/add_remove_lid/0.hdf5"
DEFAULT_MP4 = "/home/user/test/add_remove_lid/0.mp4"
DEFAULT_WEIGHTS = str(DROID_ROOT / "droid.pth")
DEFAULT_OUTPUT_ROOT = DROID_ROOT / "egodex_outputs"

TUM_FMT = "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f"
OPENCV_FROM_ARKIT_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])

GT_COLOR = [0, 200, 255]
DROID_COLOR = [255, 90, 90]
DROID_SCALE_COLOR = [255, 220, 0]
AXIS_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]


def change_pose_basis(pose_matrix, target_from_source):
    return target_from_source @ pose_matrix @ np.linalg.inv(target_from_source)


def pose_matrix_to_tum_row(timestamp_s, pose_matrix):
    quaternion = R.from_matrix(pose_matrix[:3, :3]).as_quat()
    quaternion /= np.linalg.norm(quaternion)
    return [float(timestamp_s), *pose_matrix[:3, 3].tolist(), *quaternion.tolist()]


def tum_row_to_pose_matrix(row):
    pose_matrix = np.eye(4, dtype=np.float64)
    pose_matrix[:3, 3] = row[1:4]
    pose_matrix[:3, :3] = R.from_quat(row[4:8]).as_matrix()
    return pose_matrix


def is_valid_row(row):
    return row is not None and np.isfinite(np.asarray(row[1:8], dtype=np.float64)).all()


def normalize_quaternions(quaternions):
    quaternions = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    quaternions[valid] /= norms[valid]
    return quaternions


def save_tum(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, np.asarray(rows, dtype=np.float64), fmt=TUM_FMT)
    return output_path


def load_camera_poses(root):
    camera_node = root["transforms"]["camera"]
    if isinstance(camera_node, h5py.Dataset):
        return camera_node[:].astype(np.float64)

    for key in ("matrix", "transform", "poses"):
        if key in camera_node:
            return camera_node[key][:].astype(np.float64)

    data_keys = [key for key in camera_node.keys() if "time" not in key.lower()]
    if not data_keys:
        raise ValueError("transforms/camera 中没有可用的 pose dataset")
    return camera_node[data_keys[0]][:].astype(np.float64)


def pose_data_to_matrix(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape == (4, 4):
        return pose

    if pose.size == 7:
        pose_matrix = np.eye(4, dtype=np.float64)
        pose_matrix[:3, 3] = pose[:3]
        pose_matrix[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
        return pose_matrix

    raise ValueError(f"无法解析 EgoDex 位姿维度: {pose.shape}")


def load_egodex_metadata(hdf5_path):
    with h5py.File(hdf5_path, "r") as root:
        intrinsic = root["camera/intrinsic"][:].astype(np.float64)
        if intrinsic.shape != (3, 3):
            intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)

        pose_data = load_camera_poses(root)
        world_from_camera_arkit = np.stack([pose_data_to_matrix(pose) for pose in pose_data])

    return intrinsic, world_from_camera_arkit


def egodex_world_tum(world_from_camera_arkit, frame_ids, fps):
    rows = []
    for frame_id in frame_ids:
        rows.append(pose_matrix_to_tum_row(frame_id / fps, world_from_camera_arkit[frame_id]))
    return np.asarray(rows, dtype=np.float64)


def egodex_relative_opencv_tum(world_from_camera_arkit, frame_ids, fps, origin_frame_id):
    camera_origin_from_world = np.linalg.inv(world_from_camera_arkit[origin_frame_id])
    rows = []

    for frame_id in frame_ids:
        camera_origin_from_camera_arkit = camera_origin_from_world @ world_from_camera_arkit[frame_id]
        camera_origin_from_camera_opencv = change_pose_basis(
            camera_origin_from_camera_arkit,
            OPENCV_FROM_ARKIT_CAMERA,
        )
        rows.append(pose_matrix_to_tum_row(frame_id / fps, camera_origin_from_camera_opencv))

    return np.asarray(rows, dtype=np.float64)


def relative_opencv_rows_to_arkit_world_tum(relative_opencv_rows, world_from_camera_origin_arkit):
    rows = []
    for row in np.asarray(relative_opencv_rows, dtype=np.float64):
        if not is_valid_row(row):
            continue
        camera_origin_from_camera_opencv = tum_row_to_pose_matrix(row)
        camera_origin_from_camera_arkit = change_pose_basis(
            camera_origin_from_camera_opencv,
            OPENCV_FROM_ARKIT_CAMERA,
        )
        world_from_camera_arkit = world_from_camera_origin_arkit @ camera_origin_from_camera_arkit
        rows.append(pose_matrix_to_tum_row(row[0], world_from_camera_arkit))
    return np.asarray(rows, dtype=np.float64)


def convert_tum_basis(rows, target_from_source):
    converted = []
    for row in np.asarray(rows, dtype=np.float64):
        if not is_valid_row(row):
            continue
        converted_pose = change_pose_basis(tum_row_to_pose_matrix(row), target_from_source)
        converted.append(pose_matrix_to_tum_row(row[0], converted_pose))
    return np.asarray(converted, dtype=np.float64)


def droid_trajectory_to_relative_tum(traj_est, frame_ids, fps):
    traj_est = np.asarray(traj_est, dtype=np.float64)
    if traj_est.ndim != 2 or traj_est.shape[1] != 7:
        raise ValueError(f"DROID 输出应为 N x 7，实际为 {traj_est.shape}")
    if len(traj_est) != len(frame_ids):
        raise ValueError(f"DROID 输出帧数 {len(traj_est)} 与输入帧数 {len(frame_ids)} 不一致")

    quaternions = normalize_quaternions(traj_est[:, 3:7])
    rows = []
    for pose, quaternion, frame_id in zip(traj_est, quaternions, frame_ids):
        rows.append([frame_id / fps, *pose[:3].tolist(), *quaternion.tolist()])
    return np.asarray(rows, dtype=np.float64)


def read_video_info(mp4_path):
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {mp4_path}")

    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def resized_shape(height, width, target_area):
    scale = np.sqrt(float(target_area) / float(height * width))
    new_height = max(8, int(height * scale))
    new_width = max(8, int(width * scale))
    new_height -= new_height % 8
    new_width -= new_width % 8
    return new_height, new_width


def egodex_image_stream(
    mp4_path,
    intrinsic,
    target_area,
    stride,
    start_frame,
    end_frame,
    max_frames,
):
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {mp4_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_id = start_frame
    yielded = 0

    while True:
        if end_frame is not None and frame_id >= end_frame:
            break
        ok, frame_bgr = cap.read()
        if not ok:
            break

        should_use = (frame_id - start_frame) % stride == 0
        if should_use:
            h0, w0 = frame_bgr.shape[:2]
            h1, w1 = resized_shape(h0, w0, target_area)
            image = cv2.resize(frame_bgr, (w1, h1), interpolation=cv2.INTER_AREA)
            image = np.ascontiguousarray(image[:h1, :w1])

            intrinsics = torch.as_tensor(
                [
                    intrinsic[0, 0] * (w1 / w0),
                    intrinsic[1, 1] * (h1 / h0),
                    intrinsic[0, 2] * (w1 / w0),
                    intrinsic[1, 2] * (h1 / h0),
                ],
                dtype=torch.float32,
            )
            image_tensor = torch.as_tensor(image).permute(2, 0, 1)
            yield frame_id, image_tensor[None], intrinsics

            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break

        frame_id += 1

    cap.release()


def run_droid(stream_data, args):
    if len(stream_data) < args.warmup:
        raise ValueError(
            f"输入帧数 {len(stream_data)} 小于 warmup={args.warmup}，"
            "请降低 --warmup 或增加帧数。"
        )

    first_image = stream_data[0][1]
    args.image_size = [int(first_image.shape[2]), int(first_image.shape[3])]
    args.stereo = False
    args.disable_vis = not args.enable_droid_vis

    droid = DroidAsync(args) if args.asynchronous else Droid(args)

    for frame_id, image, intrinsics in tqdm(stream_data, desc="DROID-SLAM"):
        droid.track(frame_id, image, intrinsics=intrinsics)

    traj_est = droid.terminate(stream_data)

    if args.reconstruction_path:
        save_reconstruction(droid, args.reconstruction_path)

    return traj_est


def save_reconstruction(droid, save_path):
    video = droid.video2 if hasattr(droid, "video2") else droid.video
    t = video.counter.value
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu(),
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, save_path)


def match_by_timestamp(reference_rows, estimate_rows, max_time_diff):
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


def rotation_error_deg(reference_rows, estimate_rows):
    reference_rot = R.from_quat(normalize_quaternions(reference_rows[:, 4:8]))
    estimate_rot = R.from_quat(normalize_quaternions(estimate_rows[:, 4:8]))
    return np.degrees((estimate_rot.inv() * reference_rot).magnitude())


def trajectory_stats(reference_rows, estimate_rows, fps):
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


def estimate_scale_only(reference_rows, estimate_rows, fps):
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


def scale_tum_translations(rows, scale):
    scaled = np.asarray(rows, dtype=np.float64).copy()
    scaled[:, 1:4] *= scale
    return scaled


def write_report(gt_opencv_rows, droid_scaled_opencv_rows, scale, output_dir, fps):
    scaled = (
        trajectory_stats(gt_opencv_rows, droid_scaled_opencv_rows, fps)
        if droid_scaled_opencv_rows is not None
        else None
    )
    report_path = Path(output_dir) / "comparison_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("EgoDex GT vs DROID-SLAM Mono\n")
        report.write("坐标系: 首帧相机为原点的 OpenCV camera basis (x right, y down, z forward)。\n")
        report.write("流程: GT ARKit world_from_camera -> GT relative OpenCV；DROID raw -> DROID relative OpenCV -> scale_only。\n")
        report.write("说明: 单目 DROID-SLAM 没有真实尺度，报告只保留 scale_only 后的 OpenCV 相对轨迹。\n\n")
        if scale is not None:
            report.write(f"scale_only={scale:.9f}\n\n")
        else:
            report.write("scale_only=None\n\n")

        if scaled is not None:
            report.write("[opencv_relative_scale_only]\n")
            report.write(
                f"frames={scaled['frames']} "
                f"trans_rmse={scaled['translation_rmse']:.6f}m "
                f"trans_mean={scaled['translation_mean']:.6f}m "
                f"trans_max={scaled['translation_max']:.6f}m "
                f"rot_rmse={scaled['rotation_rmse']:.3f}deg "
                f"rot_mean={scaled['rotation_mean']:.3f}deg "
                f"rot_max={scaled['rotation_max']:.3f}deg\n\n"
            )

    return report_path, scaled


def evo_result_stats(result):
    return {key: float(value) for key, value in result.stats.items()}


def evaluate_with_evo(reference_tum_path, estimate_tum_path, output_dir, fps):
    try:
        from evo.core import metrics, sync
        from evo.tools import file_interface
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "evo"
        print(f"缺少 evo 依赖，跳过 evo 评估: {missing_module}")
        return None, None

    reference_tum_path = Path(reference_tum_path)
    estimate_tum_path = Path(estimate_tum_path)
    evo_dir = Path(output_dir) / "evo_summary"
    evo_dir.mkdir(parents=True, exist_ok=True)

    traj_ref = file_interface.read_tum_trajectory_file(str(reference_tum_path))
    traj_est = file_interface.read_tum_trajectory_file(str(estimate_tum_path))
    max_time_diff = 0.5 / fps + 1e-6
    traj_ref, traj_est = sync.associate_trajectories(
        traj_ref,
        traj_est,
        max_diff=max_time_diff,
    )

    results = {}
    error_files = {}
    relations = (
        ("position_m", metrics.PoseRelation.translation_part),
        ("orientation_deg", metrics.PoseRelation.rotation_angle_deg),
    )

    for name, relation in relations:
        ape = metrics.APE(relation)
        ape.process_data((traj_ref, traj_est))
        result = ape.get_result()
        results[name] = {
            "title": result.info.get("title", name),
            "stats": evo_result_stats(result),
        }
        error_path = evo_dir / f"ape_{name}_errors.txt"
        np.savetxt(error_path, result.np_arrays["error_array"], fmt="%.9f")
        error_files[name] = str(error_path)

    metrics_path = evo_dir / "evo_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as report:
        report.write("evo APE: EgoDex GT vs DROID-SLAM scale_only\n")
        report.write(f"reference={reference_tum_path}\n")
        report.write(f"estimate={estimate_tum_path}\n")
        report.write(f"matched_poses={traj_ref.num_poses}\n")
        report.write(f"max_time_diff={max_time_diff:.9f}s\n\n")

        for name in ("position_m", "orientation_deg"):
            stats = results[name]["stats"]
            report.write(f"[{name}]\n")
            report.write(f"title={results[name]['title']}\n")
            report.write(
                f"rmse={stats['rmse']:.9f} "
                f"mean={stats['mean']:.9f} "
                f"median={stats['median']:.9f} "
                f"std={stats['std']:.9f} "
                f"min={stats['min']:.9f} "
                f"max={stats['max']:.9f} "
                f"sse={stats['sse']:.9f}\n"
            )
            report.write(f"errors={error_files[name]}\n\n")

    json_path = evo_dir / "evo_metrics.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            {
                "reference": str(reference_tum_path),
                "estimate": str(estimate_tum_path),
                "matched_poses": traj_ref.num_poses,
                "max_time_diff": max_time_diff,
                "results": results,
                "error_files": error_files,
            },
            json_file,
            indent=2,
            ensure_ascii=False,
        )

    return metrics_path, results


def rows_by_frame_id(rows, fps):
    lookup = {}
    if rows is None:
        return lookup
    for row in np.asarray(rows, dtype=np.float64):
        if is_valid_row(row):
            lookup[int(round(float(row[0]) * fps))] = row
    return lookup


def trajectory_points(rows):
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


def log_static_trajectory(rr, entity_path, rows, color, radius):
    points = trajectory_points(rows)
    if points is None:
        return
    rr.log(
        f"{entity_path}/trajectory",
        rr.LineStrips3D(points, colors=[color], radii=radius),
        static=True,
    )


def log_pose(rr, entity_path, row, color, axis_length):
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


def rr_image(rr, image_rgb):
    image = rr.Image(image_rgb)
    if hasattr(image, "compress"):
        return image.compress(jpeg_quality=80)
    return image


def set_rerun_time(rr, frame_id, fps):
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", frame_id)
        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds("time", frame_id / fps)
        return

    rr.set_time("frame", sequence=frame_id)
    rr.set_time("time", duration=frame_id / fps)


def init_rerun(app_id, spawn, save_rrd):
    try:
        import rerun as rr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "缺少 rerun-sdk。请在 DROID 环境中安装：\n"
            "  conda run -n droid-slam python -m pip install rerun-sdk"
        ) from exc

    rr.init(app_id, spawn=spawn and save_rrd is None)
    if save_rrd is not None:
        rr.save(str(save_rrd))

    try:
        view_coordinates = getattr(
            rr.ViewCoordinates,
            "RIGHT_HAND_Y_DOWN",
            rr.ViewCoordinates.RIGHT_HAND_Y_UP,
        )
        rr.log("world", view_coordinates, static=True)
    except Exception as exc:
        print(f"Rerun 坐标系设置失败，继续使用默认坐标系: {exc}")

    try:
        import rerun.blueprint as rrb

        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Horizontal(
                    column_shares=[0.42, 0.58],
                    contents=[
                        rrb.Spatial2DView(name="RGB", origin="world/input/image"),
                        rrb.Spatial3DView(name="GT vs DROID-SLAM", origin="world"),
                    ],
                ),
            ),
            make_active=True,
        )
    except Exception as exc:
        print(f"Rerun 布局配置失败，继续使用默认布局: {exc}")

    return rr


def visualize_results_in_rerun(
    gt_opencv_rows,
    droid_scaled_opencv_rows,
    mp4_path,
    frame_ids,
    fps,
    args,
):
    if args.no_rerun:
        return

    save_rrd = Path(args.save_rrd) if args.save_rrd else None
    if save_rrd is not None:
        save_rrd.parent.mkdir(parents=True, exist_ok=True)

    rr = init_rerun(args.rerun_app_id, args.rerun_spawn, save_rrd)
    log_static_trajectory(rr, "world/gt_opencv", gt_opencv_rows, GT_COLOR, args.rerun_trajectory_radius)
    log_static_trajectory(
        rr,
        "world/droid_opencv_scale_only",
        droid_scaled_opencv_rows,
        DROID_SCALE_COLOR,
        args.rerun_trajectory_radius,
    )

    gt_lookup = rows_by_frame_id(gt_opencv_rows, fps)
    scaled_lookup = rows_by_frame_id(droid_scaled_opencv_rows, fps)
    selected = set(frame_ids)
    last_frame = max(selected)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"无法打开视频，Rerun 只显示静态轨迹: {mp4_path}")
        return

    logged_frames = 0
    frame_id = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_id > last_frame:
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
            "world/droid_opencv_scale_only/current",
            scaled_lookup.get(frame_id),
            DROID_SCALE_COLOR,
            args.rerun_axis_length,
        )

        logged_frames += 1
        frame_id += 1

    cap.release()
    print(f"已发送 Rerun 可视化：{logged_frames} 帧")
    if save_rrd is not None:
        print(f"已保存 Rerun 记录: {save_rrd}")


def default_output_dir(hdf5_path):
    hdf5_path = Path(hdf5_path)
    return DEFAULT_OUTPUT_ROOT / f"{hdf5_path.parent.name}_{hdf5_path.stem}"


def remove_stale_unscaled_droid_outputs(output_dir):
    for filename in (
        "droid_raw_tum.txt",
        "droid_relative_opencv_tum.txt",
        "droid_arkit_world_tum.txt",
        "droid_arkit_world_scale_only_tum.txt",
    ):
        path = Path(output_dir) / filename
        if path.exists():
            path.unlink()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DROID-SLAM on an EgoDex hdf5/mp4 pair and visualize GT/results in Rerun."
    )
    parser.add_argument("--hdf5", default=DEFAULT_HDF5, help="EgoDex .hdf5 path")
    parser.add_argument("--mp4", default=DEFAULT_MP4, help="EgoDex .mp4 path")
    parser.add_argument("--output", help="output directory; default is DROID-SLAM/egodex_outputs/<sequence>")
    parser.add_argument("--fps", type=float, help="override video FPS; default reads FPS from mp4")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="exclusive end frame")
    parser.add_argument("--max-frames", type=int, help="debug limit after stride/start-frame filtering")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--target-area", type=int, default=384 * 512, help="DROID resize target area")
    parser.add_argument(
        "--droid-pose-basis",
        choices=("arkit", "opencv"),
        default="arkit",
        help="basis of DROID's returned trajectory; default arkit applies a y/z flip into OpenCV before comparison",
    )

    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=4.0)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", default="cuda")
    parser.add_argument("--backend_device", default="cuda")
    parser.add_argument("--motion_damping", type=float, default=0.5)
    parser.add_argument("--enable-droid-vis", action="store_true", help="enable DROID's OpenGL visualizer")
    parser.add_argument("--reconstruction_path", help="optional DROID reconstruction .pth output")

    parser.add_argument("--no-rerun", action="store_true", help="skip Rerun visualization")
    parser.add_argument("--skip-evo", action="store_true", help="skip evo APE evaluation")
    parser.add_argument("--rerun-app-id", default="egodex_droid_slam")
    parser.add_argument("--rerun-spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rrd", help="save a .rrd recording instead of spawning the viewer")
    parser.add_argument("--rerun-axis-length", type=float, default=0.04)
    parser.add_argument("--rerun-trajectory-radius", type=float, default=0.002)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride 必须 >= 1")
    if args.start_frame < 0:
        raise ValueError("--start-frame 必须 >= 0")

    hdf5_path = Path(args.hdf5)
    mp4_path = Path(args.mp4)
    output_dir = Path(args.output) if args.output else default_output_dir(hdf5_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_unscaled_droid_outputs(output_dir)

    intrinsic, world_from_camera_arkit = load_egodex_metadata(hdf5_path)
    video_info = read_video_info(mp4_path)
    fps = args.fps or video_info["fps"] or 30.0

    np.savetxt(output_dir / "camera_intrinsic.txt", intrinsic)
    print(f"输出目录: {output_dir}")
    print(
        f"EgoDex 视频: {video_info['frames']} 帧, {video_info['width']}x{video_info['height']}, "
        f"fps={fps:.3f}"
    )
    print(f"EgoDex GT: {len(world_from_camera_arkit)} 帧")

    stream_data = list(
        egodex_image_stream(
            mp4_path,
            intrinsic,
            args.target_area,
            args.stride,
            args.start_frame,
            args.end_frame,
            args.max_frames,
        )
    )
    if not stream_data:
        raise ValueError("没有读到可处理帧，请检查 --start-frame/--end-frame/--stride")

    frame_ids = [frame_id for frame_id, _, _ in stream_data]
    if max(frame_ids) >= len(world_from_camera_arkit):
        raise ValueError(
            f"所选最大帧 {max(frame_ids)} 超出 hdf5 GT 范围 {len(world_from_camera_arkit)}"
        )
    np.savetxt(output_dir / "frame_ids.txt", np.asarray(frame_ids, dtype=np.int64), fmt="%d")
    print(f"DROID 输入: {len(frame_ids)} 帧, frame {frame_ids[0]} -> {frame_ids[-1]}, stride={args.stride}")
    print(f"DROID 图像尺寸: {stream_data[0][1].shape[-2]}x{stream_data[0][1].shape[-1]}")

    origin_frame_id = frame_ids[0]
    gt_arkit_world_rows = egodex_world_tum(world_from_camera_arkit, frame_ids, fps)
    gt_opencv_rows = egodex_relative_opencv_tum(
        world_from_camera_arkit,
        frame_ids,
        fps,
        origin_frame_id,
    )
    gt_arkit_path = save_tum(gt_arkit_world_rows, output_dir / "gt_arkit_world_tum.txt")
    gt_opencv_path = save_tum(gt_opencv_rows, output_dir / "gt_relative_opencv_tum.txt")
    print(f"已保存 GT ARKit world TUM(仅留作参考): {gt_arkit_path}")
    print(f"已保存 GT OpenCV TUM(用于对比): {gt_opencv_path}")

    torch.multiprocessing.set_start_method("spawn", force=True)
    traj_est = run_droid(stream_data, args)

    droid_raw_rows = droid_trajectory_to_relative_tum(traj_est, frame_ids, fps)

    if args.droid_pose_basis == "arkit":
        droid_relative_rows = convert_tum_basis(droid_raw_rows, OPENCV_FROM_ARKIT_CAMERA)
        print("DROID raw 轨迹按 ARKit/OpenGL camera basis 处理，已转换到 OpenCV basis。")
    else:
        droid_relative_rows = droid_raw_rows
        print("DROID raw 轨迹按 OpenCV camera basis 处理，不做额外 basis 转换。")

    scale = estimate_scale_only(gt_opencv_rows, droid_relative_rows, fps)
    droid_scaled_relative_rows = None
    scaled_path = None
    if scale is not None:
        droid_scaled_relative_rows = scale_tum_translations(droid_relative_rows, scale)
        scaled_path = save_tum(
            droid_scaled_relative_rows,
            output_dir / "droid_relative_opencv_scale_only_tum.txt",
        )
        print(f"已保存 DROID OpenCV scale_only TUM: {scaled_path}")
        print(f"scale_only = {scale:.9f}")
    else:
        print("无法估计 scale_only，跳过尺度缩放轨迹。")

    report_path, scaled = write_report(
        gt_opencv_rows,
        droid_scaled_relative_rows,
        scale,
        output_dir,
        fps,
    )
    print(f"已保存对比报告: {report_path}")
    if scaled is not None:
        print(
            "OpenCV relative + scale_only | "
            f"平移 RMSE {scaled['translation_rmse']:.6f} m | "
            f"旋转 RMSE {scaled['rotation_rmse']:.3f} deg"
        )

    if scaled_path is not None and not args.skip_evo:
        evo_report_path, evo_results = evaluate_with_evo(
            gt_opencv_path,
            scaled_path,
            output_dir,
            fps,
        )
        if evo_results is not None:
            print(f"已保存 evo 评估: {evo_report_path}")
            print(
                "evo APE | "
                f"位置 RMSE {evo_results['position_m']['stats']['rmse']:.6f} m | "
                f"朝向 RMSE {evo_results['orientation_deg']['stats']['rmse']:.3f} deg"
            )

    visualize_results_in_rerun(
        gt_opencv_rows,
        droid_scaled_relative_rows,
        mp4_path,
        frame_ids,
        fps,
        args,
    )


if __name__ == "__main__":
    main()
