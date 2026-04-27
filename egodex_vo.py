import argparse
import os

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


# ===================== 配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EGODEX_HDF5_PATH = "/home/user/test/add_remove_lid/0.hdf5"
EGODEX_MP4_PATH = "/home/user/test/add_remove_lid/0.mp4"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "egodex_tum")
FPS = 30.0
RUN_CUVSLAM = True
ENABLE_RERUN = True
RERUN_APP_ID = "egodex_cuslam_arkit_world"
RERUN_AXIS_LENGTH = 0.04
RERUN_TRAJECTORY_RADIUS = 0.002
RERUN_IMAGE_PATH = "world/input/image"
RERUN_MASK_OVERLAY_PATH = "world/input/mask_overlay"
RERUN_MASK_PATH = "world/input/mask"
ENABLE_DYNAMIC_HAND_MASK = True
SAVE_MASK_IMAGES = True
MASK_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "masks")
MASK_JOINT_RADIUS = 36
MASK_BONE_RADIUS = 48
MASK_HULL_DILATION = 35
MASK_TEMPORAL_SMOOTHING = True

# EgoDex/ARKit camera: x-right, y-up, z-backward.
# OpenCV/cuSLAM relative output is bridged back into this ARKit basis for comparison.
OPENCV_FROM_ARKIT_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])

TUM_FMT = "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f"

MASK_TRANSFORM_KEYWORDS = ("Arm", "Forearm", "Hand", "Finger", "Thumb")
AXIS_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
GT_COLOR = [0, 200, 255]
CUSLAM_YZ_FLIP_COLOR = [255, 80, 80]
CUSLAM_SCALE_ONLY_COLOR = [255, 220, 0]

FINGER_SEGMENTS = (
    ("ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip"),
    (
        "IndexFingerMetacarpal",
        "IndexFingerKnuckle",
        "IndexFingerIntermediateBase",
        "IndexFingerIntermediateTip",
        "IndexFingerTip",
    ),
    (
        "MiddleFingerMetacarpal",
        "MiddleFingerKnuckle",
        "MiddleFingerIntermediateBase",
        "MiddleFingerIntermediateTip",
        "MiddleFingerTip",
    ),
    (
        "RingFingerMetacarpal",
        "RingFingerKnuckle",
        "RingFingerIntermediateBase",
        "RingFingerIntermediateTip",
        "RingFingerTip",
    ),
    (
        "LittleFingerMetacarpal",
        "LittleFingerKnuckle",
        "LittleFingerIntermediateBase",
        "LittleFingerIntermediateTip",
        "LittleFingerTip",
    ),
)


# ===================== TUM / pose 工具 =====================
def change_pose_basis(pose_matrix, target_from_source):
    return target_from_source @ pose_matrix @ np.linalg.inv(target_from_source)


def pose_matrix_to_tum_row(timestamp_s, pose_matrix):
    translation = pose_matrix[:3, 3]
    quaternion = R.from_matrix(pose_matrix[:3, :3]).as_quat()
    quaternion /= np.linalg.norm(quaternion)
    return [
        float(timestamp_s),
        *translation.tolist(),
        *quaternion.tolist(),
    ]


def tum_row_to_pose_matrix(row):
    pose_matrix = np.eye(4, dtype=np.float64)
    pose_matrix[:3, 3] = row[1:4]
    pose_matrix[:3, :3] = R.from_quat(row[4:8]).as_matrix()
    return pose_matrix


def save_tum(rows, output_path):
    rows = np.asarray(rows, dtype=np.float64)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, rows, fmt=TUM_FMT)
    return output_path


def load_tum(input_path):
    rows = np.loadtxt(input_path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] != 8:
        raise ValueError(f"TUM 文件必须是 8 列: {input_path}")
    return rows


def invert_tum_rows(rows):
    inverted = []
    for row in rows:
        if not is_valid_row(row):
            continue
        inverted.append(pose_matrix_to_tum_row(row[0], np.linalg.inv(tum_row_to_pose_matrix(row))))
    return np.asarray(inverted, dtype=np.float64)


def convert_tum_basis(rows, target_from_source):
    converted = []
    for row in rows:
        if not is_valid_row(row):
            continue
        converted_pose = change_pose_basis(tum_row_to_pose_matrix(row), target_from_source)
        converted.append(pose_matrix_to_tum_row(row[0], converted_pose))
    return np.asarray(converted, dtype=np.float64)


def is_valid_row(row):
    return row is not None and np.isfinite(np.asarray(row[1:8], dtype=np.float64)).all()


def is_dynamic_mask_transform(name):
    return name.startswith(("left", "right")) and any(
        keyword in name for keyword in MASK_TRANSFORM_KEYWORDS
    )


# ===================== GT: EgoDex/ARKit world TUM =====================
def load_egodex_metadata(hdf5_path):
    with h5py.File(hdf5_path, "r") as root:
        intrinsic = root["camera/intrinsic"][:].astype(np.float64)
        world_from_camera_arkit = root["transforms/camera"][:].astype(np.float64)
        dynamic_part_poses = {
            name: root["transforms"][name][:].astype(np.float64)
            for name in root["transforms"].keys()
            if is_dynamic_mask_transform(name)
        }
    return intrinsic, world_from_camera_arkit, dynamic_part_poses


def egodex_camera_poses_to_arkit_world_tum(world_from_camera_arkit, fps):
    rows = []
    for frame_id, pose in enumerate(world_from_camera_arkit):
        rows.append(pose_matrix_to_tum_row(frame_id / fps, pose))
    return np.asarray(rows, dtype=np.float64)


def egodex_camera_poses_to_relative_opencv_tum(
    world_from_camera_arkit,
    fps,
    origin_frame_id=0,
    frame_ids=None,
):
    if frame_ids is None:
        frame_ids = range(len(world_from_camera_arkit))
    camera0_from_world = np.linalg.inv(world_from_camera_arkit[origin_frame_id])
    rows = []

    for frame_id in frame_ids:
        pose = world_from_camera_arkit[frame_id]
        camera0_from_camera_arkit = camera0_from_world @ pose
        camera0_from_camera_opencv = change_pose_basis(
            camera0_from_camera_arkit,
            OPENCV_FROM_ARKIT_CAMERA,
        )
        rows.append(pose_matrix_to_tum_row(frame_id / fps, camera0_from_camera_opencv))

    return np.asarray(rows, dtype=np.float64)


def relative_opencv_rows_to_arkit_world_tum(relative_opencv_rows, world_from_camera0_arkit):
    rows = []
    for row in relative_opencv_rows:
        if not is_valid_row(row):
            continue
        camera0_from_camera_opencv = tum_row_to_pose_matrix(row)
        camera0_from_camera_arkit = change_pose_basis(
            camera0_from_camera_opencv,
            OPENCV_FROM_ARKIT_CAMERA,
        )
        world_from_camera_arkit = world_from_camera0_arkit @ camera0_from_camera_arkit
        rows.append(pose_matrix_to_tum_row(row[0], world_from_camera_arkit))
    return np.asarray(rows, dtype=np.float64)


# ===================== EgoDex 动态手部 mask =====================
def project_world_point_to_image(intrinsic, world_from_camera_arkit, point_world):
    camera_from_world = np.linalg.inv(world_from_camera_arkit)
    point_camera = (camera_from_world @ np.r_[point_world, 1.0])[:3]
    if point_camera[2] <= 1e-6:
        return None

    u = intrinsic[0, 0] * point_camera[0] / point_camera[2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * point_camera[1] / point_camera[2] + intrinsic[1, 2]
    if not np.isfinite([u, v]).all():
        return None
    return float(u), float(v)


def collect_dynamic_part_uvs(intrinsic, world_from_camera_arkit, dynamic_part_poses, frame_id, width, height):
    projected = {}
    margin = max(width, height)

    for name, poses in dynamic_part_poses.items():
        if frame_id >= len(poses):
            continue
        uv = project_world_point_to_image(
            intrinsic,
            world_from_camera_arkit,
            poses[frame_id][:3, 3],
        )
        if uv is None:
            continue

        u, v = uv
        if -margin <= u <= width + margin and -margin <= v <= height + margin:
            projected[name] = (int(round(u)), int(round(v)))

    return projected


def draw_mask_chain(mask, projected, names):
    points = [projected[name] for name in names if name in projected]
    if len(points) < 2:
        return

    for start, end in zip(points[:-1], points[1:]):
        cv2.line(mask, start, end, 255, thickness=MASK_BONE_RADIUS, lineType=cv2.LINE_AA)


def draw_side_dynamic_mask(mask, projected, side):
    prefix = side
    arm_chain = [f"{prefix}Arm", f"{prefix}Forearm", f"{prefix}Hand"]
    draw_mask_chain(mask, projected, arm_chain)

    hand_points = []
    if f"{prefix}Hand" in projected:
        hand_points.append(projected[f"{prefix}Hand"])

    for segment in FINGER_SEGMENTS:
        chain = [f"{prefix}{name}" for name in segment]
        if f"{prefix}Hand" in projected:
            chain = [f"{prefix}Hand", *chain]
        draw_mask_chain(mask, projected, chain)
        hand_points.extend(projected[name] for name in chain if name in projected)

    for name, point in projected.items():
        if name.startswith(prefix):
            cv2.circle(mask, point, MASK_JOINT_RADIUS, 255, thickness=-1, lineType=cv2.LINE_AA)

    if len(hand_points) >= 3:
        hull = cv2.convexHull(np.asarray(hand_points, dtype=np.int32))
        cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_AA)


def create_dynamic_hand_mask(
    intrinsic,
    world_from_camera_arkit,
    dynamic_part_poses,
    frame_id,
    width,
    height,
):
    mask = np.zeros((height, width), dtype=np.uint8)
    projected = collect_dynamic_part_uvs(
        intrinsic,
        world_from_camera_arkit,
        dynamic_part_poses,
        frame_id,
        width,
        height,
    )
    if not projected:
        return mask

    draw_side_dynamic_mask(mask, projected, "left")
    draw_side_dynamic_mask(mask, projected, "right")

    if MASK_HULL_DILATION > 0:
        kernel_size = MASK_HULL_DILATION * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel)

    return np.ascontiguousarray(mask)


# ===================== cuSLAM: MP4 -> OpenCV TUM =====================
def create_cuvslam_tracker(intrinsic, image_size):
    import cuvslam

    camera = cuvslam.Camera()
    camera.size = image_size
    camera.principal = [float(intrinsic[0, 2]), float(intrinsic[1, 2])]
    camera.focal = [float(intrinsic[0, 0]), float(intrinsic[1, 1])]

    # EgoDex image edge features are often noisy after video compression.
    camera.border_top = 20
    camera.border_bottom = 20
    camera.border_left = 20
    camera.border_right = 20

    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=False,
        odometry_mode=cuvslam.Tracker.OdometryMode.Mono,
    )
    return cuvslam.Tracker(cuvslam.Rig([camera]), cfg)


def run_cuvslam_mono_to_tum(
    mp4_path,
    intrinsic,
    fps,
    world_from_camera_arkit=None,
    dynamic_part_poses=None,
    start_frame=0,
    end_frame=None,
    stride=1,
    max_frames=None,
):
    try:
        import cuvslam  # noqa: F401
    except ModuleNotFoundError:
        print("未找到 Python 模块 cuvslam，跳过 cuSLAM TUM 生成。")
        print("请在安装了 PyCuVSLAM 的环境里运行这个脚本。")
        return None

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {mp4_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = create_cuvslam_tracker(intrinsic, (width, height))
    use_dynamic_mask = (
        ENABLE_DYNAMIC_HAND_MASK
        and world_from_camera_arkit is not None
        and dynamic_part_poses
    )
    if use_dynamic_mask:
        print(f"启用 EgoDex 手/臂动态 mask，共 {len(dynamic_part_poses)} 个动态 transform")
        if SAVE_MASK_IMAGES:
            os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
    else:
        print("未启用动态 mask。")

    rows = []
    failed_frames = []
    mask_coverages = []
    prev_mask = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_id = start_frame
    yielded_frames = 0
    while True:
        if end_frame is not None and frame_id >= end_frame:
            break
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if (frame_id - start_frame) % stride != 0:
            frame_id += 1
            continue

        timestamp_ns = int(round(frame_id / fps * 1e9))
        frame_bgr = np.ascontiguousarray(frame_bgr)

        masks = None
        if use_dynamic_mask and frame_id < len(world_from_camera_arkit):
            mask = create_dynamic_hand_mask(
                intrinsic,
                world_from_camera_arkit[frame_id],
                dynamic_part_poses,
                frame_id,
                width,
                height,
            )
            if MASK_TEMPORAL_SMOOTHING and prev_mask is not None:
                tracker_mask = cv2.bitwise_or(mask, prev_mask)
            else:
                tracker_mask = mask
            prev_mask = mask

            if SAVE_MASK_IMAGES:
                cv2.imwrite(os.path.join(MASK_OUTPUT_DIR, f"{frame_id:06d}.png"), tracker_mask)
            masks = [tracker_mask]
            mask_coverages.append(float(np.count_nonzero(tracker_mask)) / float(width * height))

        pose_estimate, _ = tracker.track(timestamp_ns, [frame_bgr], masks)

        if pose_estimate.world_from_rig is None:
            failed_frames.append(frame_id)
            frame_id += 1
            continue

        # Raw PyCuVSLAM world_from_rig. The EgoDex GT comparison basis is applied later.
        pose = pose_estimate.world_from_rig.pose
        row = [
            frame_id / fps,
            *np.asarray(pose.translation, dtype=np.float64).tolist(),
            *np.asarray(pose.rotation, dtype=np.float64).tolist(),
        ]
        rows.append(row)
        frame_id += 1
        yielded_frames += 1
        if max_frames is not None and yielded_frames >= max_frames:
            break

    cap.release()
    print(f"cuSLAM 跟踪完成：有效 {len(rows)} 帧，失败 {len(failed_frames)} 帧")
    if mask_coverages:
        print(
            "动态 mask 覆盖率："
            f"mean={np.mean(mask_coverages):.3%}, "
            f"min={np.min(mask_coverages):.3%}, "
            f"max={np.max(mask_coverages):.3%}"
        )
    if failed_frames:
        print(f"cuSLAM 失败帧: {failed_frames}")

    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


# ===================== 对齐与诊断 =====================
def match_by_timestamp(reference_rows, estimate_rows, max_time_diff=1e-4):
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


def normalize_quaternions(quaternions):
    quaternions = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    good = norms[:, 0] > 1e-12
    quaternions[good] /= norms[good]
    return quaternions


def rotation_error_deg(reference_rows, estimate_rows):
    reference_rot = R.from_quat(normalize_quaternions(reference_rows[:, 4:8]))
    estimate_rot = R.from_quat(normalize_quaternions(estimate_rows[:, 4:8]))
    return np.degrees((estimate_rot.inv() * reference_rot).magnitude())


def trajectory_stats(reference_rows, estimate_rows):
    ref, est = match_by_timestamp(reference_rows, estimate_rows)
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
        "translation_rmse": float(np.sqrt(np.mean(translation_errors ** 2))),
        "translation_mean": float(np.mean(translation_errors)),
        "translation_max": float(np.max(translation_errors)),
        "rotation_rmse": float(np.sqrt(np.mean(rotation_errors ** 2))),
        "rotation_mean": float(np.mean(rotation_errors)),
        "rotation_max": float(np.max(rotation_errors)),
    }


def estimate_scale_only(reference_rows, estimate_rows):
    ref, est = match_by_timestamp(reference_rows, estimate_rows)
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


def align_cuslam_to_relative_opencv_basis(cuslam_rows):
    # This produces the same first-camera-relative OpenCV basis used internally
    # for scale estimation before anchoring the trajectory in ARKit world.
    return convert_tum_basis(cuslam_rows, OPENCV_FROM_ARKIT_CAMERA)


def write_comparison_report(gt_rows, cuslam_arkit_world_rows, cuslam_scaled_rows, scale, output_dir):
    direct = trajectory_stats(gt_rows, cuslam_arkit_world_rows)
    scaled = trajectory_stats(gt_rows, cuslam_scaled_rows)
    report_path = os.path.join(output_dir, "comparison_report.txt")

    with open(report_path, "w", encoding="utf-8") as report:
        report.write("EgoDex GT vs cuSLAM Mono\n")
        report.write("坐标系: 原始 EgoDex/ARKit world_from_camera。\n")
        report.write("流程: EgoDex 动态手/臂 mask -> cuSLAM Mono -> 相对轨迹转回 ARKit basis -> 接到 GT 第 0 帧 world pose。\n")
        report.write("说明: Mono 没有真实尺度，scale_only 只缩放第 0 帧相对平移，再放回 ARKit world 用于比较轨迹形状。\n\n")
        report.write(f"scale_only={scale:.9f}\n\n")

        if direct is not None:
            report.write("[arkit_world]\n")
            report.write(
                f"frames={direct['frames']} "
                f"trans_rmse={direct['translation_rmse']:.6f}m "
                f"trans_mean={direct['translation_mean']:.6f}m "
                f"trans_max={direct['translation_max']:.6f}m "
                f"rot_rmse={direct['rotation_rmse']:.3f}deg "
                f"rot_mean={direct['rotation_mean']:.3f}deg "
                f"rot_max={direct['rotation_max']:.3f}deg\n\n"
            )

        if scaled is not None:
            report.write("[arkit_world_scale_only]\n")
            report.write(
                f"frames={scaled['frames']} "
                f"trans_rmse={scaled['translation_rmse']:.6f}m "
                f"trans_mean={scaled['translation_mean']:.6f}m "
                f"trans_max={scaled['translation_max']:.6f}m "
                f"rot_rmse={scaled['rotation_rmse']:.3f}deg "
                f"rot_mean={scaled['rotation_mean']:.3f}deg "
                f"rot_max={scaled['rotation_max']:.3f}deg\n"
            )

    return report_path, direct, scaled


# ===================== Rerun 可视化 =====================
def rows_by_frame_id(rows, fps):
    if rows is None:
        return {}

    lookup = {}
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


def log_static_trajectory_to_rerun(rr, entity_path, rows, color):
    points = trajectory_points(rows)
    if points is None:
        return

    rr.log(
        f"{entity_path}/trajectory",
        rr.LineStrips3D(
            points,
            colors=[color],
            radii=RERUN_TRAJECTORY_RADIUS,
        ),
        static=True,
    )


def log_tum_pose_to_rerun(rr, entity_path, row, color):
    if row is None or not is_valid_row(row):
        return

    quaternion = normalize_quaternions(np.asarray(row[4:8], dtype=np.float64)[None, :])[0]
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=np.asarray(row[1:4], dtype=np.float64),
            quaternion=quaternion,
        ),
        rr.Arrows3D(
            vectors=np.eye(3) * RERUN_AXIS_LENGTH,
            colors=AXIS_COLORS,
            radii=RERUN_AXIS_LENGTH * 0.05,
        ),
    )
    rr.log(
        f"{entity_path}/origin",
        rr.Points3D(
            [[0.0, 0.0, 0.0]],
            colors=[color],
            radii=RERUN_AXIS_LENGTH * 0.1,
        ),
    )


def create_mask_overlay(image_rgb, mask):
    overlay = image_rgb.copy()
    masked = mask > 0
    if np.any(masked):
        red = np.array([255, 0, 0], dtype=np.float32)
        overlay[masked] = (0.45 * overlay[masked].astype(np.float32) + 0.55 * red).astype(np.uint8)
    return overlay


def init_rerun_visualizer():
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ModuleNotFoundError:
        print("未找到 rerun-sdk，跳过 Rerun 可视化。可用 `pip install rerun-sdk` 安装。")
        return None

    try:
        rr.init(RERUN_APP_ID, spawn=True)
        view_coordinates = getattr(
            rr.ViewCoordinates,
            "RIGHT_HAND_Y_UP",
            rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
        )
        rr.log("world", view_coordinates, static=True)
    except Exception as exc:
        print(f"Rerun 初始化失败，跳过可视化: {exc}")
        return None

    try:
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Horizontal(
                    column_shares=[0.42, 0.58],
                    contents=[
                        rrb.Vertical(
                            contents=[
                                rrb.Spatial2DView(name="RGB", origin=RERUN_IMAGE_PATH),
                                rrb.Spatial2DView(name="Mask Overlay", origin=RERUN_MASK_OVERLAY_PATH),
                                rrb.Spatial2DView(name="Mask", origin=RERUN_MASK_PATH),
                            ]
                        ),
                        rrb.Spatial3DView(name="GT vs cuSLAM", origin="world"),
                    ],
                ),
            ),
            make_active=True,
        )
    except Exception as exc:
        print(f"Rerun 布局配置失败，继续使用默认布局: {exc}")

    return rr


def visualize_results_in_rerun(
    gt_rows,
    cuslam_arkit_world_rows,
    cuslam_scale_only_rows,
    mp4_path,
    output_dir,
    fps,
):
    if not ENABLE_RERUN:
        return

    rr = init_rerun_visualizer()
    if rr is None:
        return

    log_static_trajectory_to_rerun(rr, "world/gt", gt_rows, GT_COLOR)
    log_static_trajectory_to_rerun(
        rr,
        "world/cuslam_arkit_world",
        cuslam_arkit_world_rows,
        CUSLAM_YZ_FLIP_COLOR,
    )
    log_static_trajectory_to_rerun(
        rr,
        "world/cuslam_scale_only",
        cuslam_scale_only_rows,
        CUSLAM_SCALE_ONLY_COLOR,
    )

    gt_lookup = rows_by_frame_id(gt_rows, fps)
    arkit_world_lookup = rows_by_frame_id(cuslam_arkit_world_rows, fps)
    scale_only_lookup = rows_by_frame_id(cuslam_scale_only_rows, fps)

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"无法打开视频，Rerun 只显示静态轨迹: {mp4_path}")
        return

    mask_dir = MASK_OUTPUT_DIR if os.path.isdir(MASK_OUTPUT_DIR) else os.path.join(output_dir, "masks")
    frame_id = 0
    logged_frames = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        rr.set_time_sequence("frame", frame_id)
        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds("time", frame_id / fps)

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rr.log(RERUN_IMAGE_PATH, rr.Image(image_rgb).compress(jpeg_quality=80))

        mask_path = os.path.join(mask_dir, f"{frame_id:06d}.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                rr.log(RERUN_MASK_PATH, rr.Image(mask))
                rr.log(
                    RERUN_MASK_OVERLAY_PATH,
                    rr.Image(create_mask_overlay(image_rgb, mask)).compress(jpeg_quality=80),
                )

        log_tum_pose_to_rerun(rr, "world/gt/current", gt_lookup.get(frame_id), GT_COLOR)
        log_tum_pose_to_rerun(
            rr,
            "world/cuslam_arkit_world/current",
            arkit_world_lookup.get(frame_id),
            CUSLAM_YZ_FLIP_COLOR,
        )
        log_tum_pose_to_rerun(
            rr,
            "world/cuslam_scale_only/current",
            scale_only_lookup.get(frame_id),
            CUSLAM_SCALE_ONLY_COLOR,
        )

        frame_id += 1
        logged_frames += 1

    cap.release()
    print(f"已发送 Rerun 可视化：{logged_frames} 帧")


def remove_legacy_outputs(output_dir):
    for filename in [
        "coordinate_diagnosis.txt",
        "gt_camera_opencv_tum.txt",
        "cuslam_world_from_camera_opencv_tum.txt",
        "cuslam_yz_flip_tum.txt",
        "cuslam_yz_flip_scale_only_tum.txt",
        "cuslam_coordinate_aligned_tum.txt",
        "cuslam_coordinate_aligned_scale_only_tum.txt",
        "cuslam_translation_shape_sim3_tum.txt",
        "cuslam_best_sim3_to_gt_tum.txt",
    ]:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            os.remove(path)

    if os.path.isdir(MASK_OUTPUT_DIR):
        for entry in os.scandir(MASK_OUTPUT_DIR):
            if entry.is_file() and entry.name.endswith(".png"):
                os.remove(entry.path)


# ===================== 主流程 =====================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cuVSLAM mono on an EgoDex hdf5/mp4 pair and export TUM trajectories."
    )
    parser.add_argument("--hdf5", default=EGODEX_HDF5_PATH, help="EgoDex .hdf5 path")
    parser.add_argument("--mp4", default=EGODEX_MP4_PATH, help="EgoDex .mp4 path")
    parser.add_argument("--output", default=OUTPUT_DIR, help="output directory")
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="exclusive end frame")
    parser.add_argument("--max-frames", type=int, help="debug limit after stride/start-frame filtering")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--no-rerun", action="store_true", help="skip per-method Rerun visualization")
    parser.add_argument("--no-dynamic-mask", action="store_true", help="disable EgoDex hand/arm dynamic mask")
    parser.add_argument("--no-save-mask-images", action="store_true", help="do not write mask images")
    parser.add_argument("--rerun-app-id", default=RERUN_APP_ID)
    parser.add_argument("--rerun-axis-length", type=float, default=RERUN_AXIS_LENGTH)
    parser.add_argument("--rerun-trajectory-radius", type=float, default=RERUN_TRAJECTORY_RADIUS)
    return parser.parse_args()


def main():
    global ENABLE_RERUN
    global ENABLE_DYNAMIC_HAND_MASK
    global MASK_OUTPUT_DIR
    global RERUN_APP_ID
    global RERUN_AXIS_LENGTH
    global RERUN_TRAJECTORY_RADIUS
    global SAVE_MASK_IMAGES

    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride 必须 >= 1")
    if args.start_frame < 0:
        raise ValueError("--start-frame 必须 >= 0")

    output_dir = os.path.abspath(os.path.expanduser(args.output))
    hdf5_path = os.path.abspath(os.path.expanduser(args.hdf5))
    mp4_path = os.path.abspath(os.path.expanduser(args.mp4))
    fps = float(args.fps)

    ENABLE_RERUN = not args.no_rerun
    ENABLE_DYNAMIC_HAND_MASK = not args.no_dynamic_mask
    SAVE_MASK_IMAGES = not args.no_save_mask_images
    MASK_OUTPUT_DIR = os.path.join(output_dir, "masks")
    RERUN_APP_ID = args.rerun_app_id
    RERUN_AXIS_LENGTH = args.rerun_axis_length
    RERUN_TRAJECTORY_RADIUS = args.rerun_trajectory_radius

    os.makedirs(output_dir, exist_ok=True)
    remove_legacy_outputs(output_dir)
    print(f"所有输出都会写入: {output_dir}")

    intrinsic, world_from_camera_arkit, dynamic_part_poses = load_egodex_metadata(hdf5_path)
    np.savetxt(os.path.join(output_dir, "camera_intrinsic.txt"), intrinsic)

    gt_rows = egodex_camera_poses_to_arkit_world_tum(world_from_camera_arkit, fps)
    gt_path = save_tum(gt_rows, os.path.join(output_dir, "gt_camera_arkit_world_tum.txt"))
    print(f"已保存 GT TUM(原始 ARKit world): {gt_path}")

    if not RUN_CUVSLAM:
        return

    cuslam_rows = run_cuvslam_mono_to_tum(
        mp4_path,
        intrinsic,
        fps,
        world_from_camera_arkit,
        dynamic_part_poses,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    if cuslam_rows is None:
        print("只生成了 GT TUM；cuSLAM TUM 未生成。")
        return

    origin_frame_id = int(round(float(cuslam_rows[0, 0]) * fps))
    frame_ids = [int(round(float(row[0]) * fps)) for row in cuslam_rows]
    gt_relative_opencv_rows = egodex_camera_poses_to_relative_opencv_tum(
        world_from_camera_arkit,
        fps,
        origin_frame_id=origin_frame_id,
        frame_ids=frame_ids,
    )
    cuslam_relative_opencv_rows = align_cuslam_to_relative_opencv_basis(cuslam_rows)
    cuslam_arkit_world_rows = relative_opencv_rows_to_arkit_world_tum(
        cuslam_relative_opencv_rows,
        world_from_camera_arkit[origin_frame_id],
    )
    cuslam_path = save_tum(
        cuslam_arkit_world_rows,
        os.path.join(output_dir, "cuslam_arkit_world_tum.txt"),
    )
    print(f"已保存 cuSLAM TUM(接回 ARKit world): {cuslam_path}")

    scale = estimate_scale_only(gt_relative_opencv_rows, cuslam_relative_opencv_rows)
    if scale is None:
        print("无法估计 scale_only，跳过尺度对齐 TUM。")
        visualize_results_in_rerun(
            gt_rows,
            cuslam_arkit_world_rows,
            None,
            mp4_path,
            output_dir,
            fps,
        )
        return

    cuslam_relative_scale_only_rows = scale_tum_translations(cuslam_relative_opencv_rows, scale)
    cuslam_scale_only_rows = relative_opencv_rows_to_arkit_world_tum(
        cuslam_relative_scale_only_rows,
        world_from_camera_arkit[origin_frame_id],
    )
    scale_only_path = save_tum(
        cuslam_scale_only_rows,
        os.path.join(output_dir, "cuslam_arkit_world_scale_only_tum.txt"),
    )
    print(f"已保存 ARKit world + 单尺度缩放 TUM: {scale_only_path}")
    print(f"scale_only = {scale:.9f}")

    report_path, direct, scaled = write_comparison_report(
        gt_rows,
        cuslam_arkit_world_rows,
        cuslam_scale_only_rows,
        scale,
        output_dir,
    )
    print(f"已保存对比报告: {report_path}")
    if direct is not None:
        print(
            "ARKit world 直接对比 | "
            f"平移 RMSE {direct['translation_rmse']:.6f} m | "
            f"旋转 RMSE {direct['rotation_rmse']:.3f} deg"
        )
    if scaled is not None:
        print(
            "ARKit world + scale_only | "
            f"平移 RMSE {scaled['translation_rmse']:.6f} m | "
            f"旋转 RMSE {scaled['rotation_rmse']:.3f} deg"
        )

    visualize_results_in_rerun(
        gt_rows,
        cuslam_arkit_world_rows,
        cuslam_scale_only_rows,
        mp4_path,
        output_dir,
        fps,
    )


if __name__ == "__main__":
    main()
