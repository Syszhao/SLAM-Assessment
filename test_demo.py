# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test camera tracking on a single scene."""

# pylint: disable=invalid-name
# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
# pylint: disable=redefined-outer-name
# pylint: disable=undefined-variable
# pylint: disable=undefined-loop-variable

import sys

sys.path.append("base/droid_slam")

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import glob
import argparse
from lietorch import SE3

import torch.nn.functional as F
from droid import Droid


def resolve_mask_paths(mask_dir, image_list):
  if not mask_dir:
    return None
  mask_paths = []
  for index, image_file in enumerate(image_list):
    basename = os.path.basename(image_file)
    stem = os.path.splitext(basename)[0]
    candidates = [
        os.path.join(mask_dir, basename),
        os.path.join(mask_dir, f"{stem}.png"),
        os.path.join(mask_dir, f"{index:06d}.png"),
    ]
    mask_file = next((path for path in candidates if os.path.exists(path)), None)
    if mask_file is None:
      raise FileNotFoundError(
          f"Missing valid mask for {image_file}; tried {candidates}"
      )
    mask_paths.append(mask_file)
  return mask_paths


def load_intrinsics(intrinsics_path, image_shape, fallback_fov):
  """Load a camera matrix in source-image pixels, or fall back to UniDepth FoV."""
  h, w = image_shape[:2]
  if intrinsics_path:
    values = np.loadtxt(intrinsics_path)
    if values.shape == (3, 3):
      K = values.astype(np.float32)
    elif values.size == 4:
      fx, fy, cx, cy = values.reshape(-1).astype(np.float32)
      K = np.eye(3, dtype=np.float32)
      K[0, 0] = fx
      K[1, 1] = fy
      K[0, 2] = cx
      K[1, 2] = cy
    else:
      raise ValueError(
          "--intrinsics must point to a 3x3 matrix or fx fy cx cy values, "
          f"got shape {values.shape}"
      )
    print("************** EXTERNAL K ", K)
    return K

  print("************** UNIDEPTH FOV ", fallback_fov)
  ff = w / (2 * np.tan(np.radians(fallback_fov / 2.0)))
  K = np.eye(3, dtype=np.float32)
  K[0, 0] = ff
  K[1, 1] = ff
  K[0, 2] = w / 2.0
  K[1, 2] = h / 2.0
  return K


def load_metric_depth(depth_path):
  """Load metric depth and fill stereo holes without changing the metric scale."""
  with np.load(depth_path) as data:
    depth = np.asarray(data["depth"], dtype=np.float32)
    fov = np.asarray(data["fov"]).item() if "fov" in data else np.nan
    if "stereo_valid" in data:
      valid = np.asarray(data["stereo_valid"]).astype(bool)
      valid &= np.isfinite(depth) & (depth > 1e-4)
      if np.any(valid):
        fill_depth = np.median(depth[valid])
        depth = np.where(valid, depth, fill_depth)
      else:
        depth = np.zeros_like(depth)

  depth = np.where(np.isfinite(depth) & (depth > 1e-4), depth, 0.0)
  return depth.astype(np.float32), float(fov)


def clear_pngs(directory):
  os.makedirs(directory, exist_ok=True)
  for path in glob.glob(os.path.join(directory, "*.png")):
    os.remove(path)


def prepare_mask_vis_dirs(output_path):
  if not output_path:
    return None
  raw_dir = os.path.join(output_path, "raw")
  overlay_dir = os.path.join(output_path, "overlay")
  info_path = os.path.join(output_path, "README.txt")
  if os.path.exists(info_path):
    os.remove(info_path)
  clear_pngs(raw_dir)
  clear_pngs(overlay_dir)
  return raw_dir, overlay_dir


def image_tensor_to_bgr(image):
  image_np = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else image
  image_np = np.asarray(image_np)
  if image_np.ndim == 4 and image_np.shape[0] == 1:
    image_np = image_np[0]
  if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
    image_np = np.transpose(image_np, (1, 2, 0))
  image_np = np.squeeze(image_np)
  if image_np.ndim == 2:
    image_np = cv2.cvtColor(np.uint8(np.clip(image_np, 0, 255)), cv2.COLOR_GRAY2BGR)
  return np.uint8(np.clip(image_np, 0, 255))


def unit_mask_to_uint8(mask):
  mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
  mask_np = np.squeeze(np.asarray(mask_np, dtype=np.float32))
  if mask_np.ndim > 2:
    mask_np = mask_np[..., 0]
  mask_np = np.nan_to_num(mask_np, nan=0.0, posinf=1.0, neginf=0.0)
  return np.uint8(np.clip(mask_np, 0.0, 1.0) * 255.0)


def mask_values_to_uint8(mask):
  mask_np = np.squeeze(np.asarray(mask, dtype=np.float32))
  if mask_np.ndim > 2:
    mask_np = mask_np[..., 0]
  mask_np = np.nan_to_num(mask_np, nan=0.0, posinf=0.0, neginf=0.0)
  min_value = float(np.min(mask_np)) if mask_np.size else 0.0
  max_value = float(np.max(mask_np)) if mask_np.size else 0.0
  if min_value < 0.0 or max_value > 1.0:
    span = max(max_value - min_value, 1e-6)
    mask_np = (mask_np - min_value) / span
  return np.uint8(np.clip(mask_np, 0.0, 1.0) * 255.0)


def save_mask_visualization(mask_vis_dirs, t, image, mask):
  if mask_vis_dirs is None:
    return
  raw_dir, overlay_dir = mask_vis_dirs
  image_bgr = image_tensor_to_bgr(image)
  mask_u8 = unit_mask_to_uint8(mask)
  if mask_u8.shape[:2] != image_bgr.shape[:2]:
    mask_u8 = cv2.resize(
        mask_u8,
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
  heatmap = cv2.applyColorMap(mask_u8, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
  overlay = cv2.addWeighted(image_bgr, 0.65, heatmap, 0.35, 0.0)
  cv2.imwrite(os.path.join(raw_dir, f"{t:06d}.png"), mask_u8)
  cv2.imwrite(os.path.join(overlay_dir, f"{t:06d}.png"), overlay)


def save_motion_mask_visualizations(output_path, images, motion_prob):
  if not output_path:
    return
  mask_vis_dirs = prepare_mask_vis_dirs(output_path)
  os.makedirs(output_path, exist_ok=True)
  motion_prob = np.asarray(motion_prob)
  if motion_prob.ndim < 3:
    info_path = os.path.join(output_path, "README.txt")
    with open(info_path, "w", encoding="utf-8") as file:
      file.write(
          "Pixel-wise Mega-SAM motion mask is unavailable. "
          "Run without --disable_full_ba to produce it.\n"
      )
    print(f"WARNING: motion mask visualization unavailable; wrote {info_path}")
    return

  raw_dir, overlay_dir = mask_vis_dirs
  count = min(len(images), motion_prob.shape[0])
  for index in range(count):
    image_bgr = image_tensor_to_bgr(images[index])
    mask_u8 = mask_values_to_uint8(motion_prob[index])
    if mask_u8.shape[:2] != image_bgr.shape[:2]:
      mask_u8 = cv2.resize(
          mask_u8,
          (image_bgr.shape[1], image_bgr.shape[0]),
          interpolation=cv2.INTER_LINEAR,
      )
    heatmap = cv2.applyColorMap(mask_u8, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
    overlay = cv2.addWeighted(image_bgr, 0.65, heatmap, 0.35, 0.0)
    cv2.imwrite(os.path.join(raw_dir, f"{index:06d}.png"), mask_u8)
    cv2.imwrite(os.path.join(overlay_dir, f"{index:06d}.png"), overlay)


def image_stream(
    image_list,
    depth_list,
    scene_name,
    use_depth=False,
    aligns=None,
    K=None,
    depth_source="mono_aligned",
    stride=1,
    valid_mask_paths=None,
):
  """image generator."""
  del scene_name, stride

  fx, fy, cx, cy = (
      K[0, 0],
      K[1, 1],
      K[0, 2],
      K[1, 2],
  )  # np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()

  for t, (image_file) in enumerate(image_list):
    image = cv2.imread(image_file)
    if image is None:
      raise FileNotFoundError(f"Failed to read image: {image_file}")

    if depth_source == "metric":
      depth = np.asarray(depth_list[t], dtype=np.float32)
      depth = np.where(np.isfinite(depth) & (depth > 1e-4), depth, 0.0)
    elif depth_source == "mono_aligned":
      if aligns is None:
        raise ValueError("mono_aligned depth_source requires aligns")
      mono_disp = depth_list[t]
      depth = np.clip(
          1.0 / ((1.0 / aligns[2]) * (aligns[0] * mono_disp + aligns[1])),
          1e-4,
          1e4,
      )
      depth[depth < 1e-2] = 0.0
    else:
      raise ValueError(f"Unsupported depth_source: {depth_source}")

    # breakpoint()
    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
    image = image[: h1 - h1 % 8, : w1 - w1 % 8]

    # if t == 4 or t == 29:
    # imageio.imwrite("debug/camel_%d.png"%t, image[..., ::-1])

    image = torch.as_tensor(image).permute(2, 0, 1)
    # print("image ", image.shape)
    # breakpoint()

    depth = torch.as_tensor(depth)
    depth = F.interpolate(
        depth[None, None], (h1, w1), mode="nearest-exact"
    ).squeeze()
    depth = depth[: h1 - h1 % 8, : w1 - w1 % 8]

    if valid_mask_paths is None:
      mask = torch.ones_like(depth)
    else:
      valid_mask = cv2.imread(valid_mask_paths[t], cv2.IMREAD_GRAYSCALE)
      if valid_mask is None:
        raise FileNotFoundError(f"Failed to read valid mask: {valid_mask_paths[t]}")
      valid_mask = cv2.resize(
          valid_mask, (w1, h1), interpolation=cv2.INTER_NEAREST
      )
      valid_mask = valid_mask[: h1 - h1 % 8, : w1 - w1 % 8]
      mask = torch.as_tensor(valid_mask.astype(np.float32) / 255.0)

    intrinsics = torch.as_tensor([fx, fy, cx, cy])
    intrinsics[0::2] *= w1 / w0
    intrinsics[1::2] *= h1 / h0

    if use_depth:
      yield t, image[None], depth, intrinsics, mask
    else:
      yield t, image[None], intrinsics, mask


def save_full_reconstruction(
    droid, full_traj, rgb_list, senor_depth_list, motion_prob, scene_name
):
  """Save full reconstruction."""
  from pathlib import Path
  t = full_traj.shape[0]
  images = np.array(rgb_list[:t])  # droid.video.images[:t].cpu().numpy()
  disps = 1.0 / (np.array(senor_depth_list[:t]) + 1e-6)

  poses = full_traj  # .cpu().numpy()
  intrinsics = droid.video.intrinsics[:t].cpu().numpy()

  Path("reconstructions/{}".format(scene_name)).mkdir(
      parents=True, exist_ok=True
  )
  np.save("reconstructions/{}/images.npy".format(scene_name), images)
  np.save("reconstructions/{}/disps.npy".format(scene_name), disps)
  np.save("reconstructions/{}/poses.npy".format(scene_name), poses)
  np.save(
      "reconstructions/{}/intrinsics.npy".format(scene_name), intrinsics * 8.0
  )
  np.save("reconstructions/{}/motion_prob.npy".format(scene_name), motion_prob)

  intrinsics = intrinsics[0] * 8.0
  poses_th = torch.as_tensor(poses, device="cpu")
  cam_c2w = SE3(poses_th).inv().matrix().numpy()

  K = np.eye(3)
  K[0, 0] = intrinsics[0]
  K[1, 1] = intrinsics[1]
  K[0, 2] = intrinsics[2]
  K[1, 2] = intrinsics[3]
  print("K ", K)
  print("img_data ", images.shape)
  print("disp_data ", disps.shape)

  max_preview_frames = min(1000, images.shape[0])
  print("outputs/%s_droid.npz" % scene_name)
  Path("outputs").mkdir(parents=True, exist_ok=True)

  np.savez(
      "outputs/%s_droid.npz" % scene_name,
      images=np.uint8(
          images[:max_preview_frames, ::-1, ...].transpose(0, 2, 3, 1)
      ),
      depths=np.float32(1.0 / disps[:max_preview_frames, ...]),
      intrinsic=K,
      cam_c2w=cam_c2w,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--datapath")
  parser.add_argument("--weights", default="droid.pth")
  parser.add_argument("--buffer", type=int, default=1024)
  parser.add_argument("--image_size", default=[240, 320])
  parser.add_argument("--disable_vis", action="store_true")

  parser.add_argument("--beta", type=float, default=0.3)
  parser.add_argument(
      "--filter_thresh", type=float, default=2.0
  )  # motion threhold for keyframe
  parser.add_argument("--warmup", type=int, default=8)
  parser.add_argument("--keyframe_thresh", type=float, default=2.0)
  parser.add_argument("--frontend_thresh", type=float, default=12.0)
  parser.add_argument("--frontend_window", type=int, default=25)
  parser.add_argument("--frontend_radius", type=int, default=2)
  parser.add_argument("--frontend_nms", type=int, default=1)

  parser.add_argument("--stereo", action="store_true")
  parser.add_argument("--depth", action="store_true")
  parser.add_argument("--upsample", action="store_true")
  parser.add_argument("--scene_name", help="scene_name")

  parser.add_argument("--backend_thresh", type=float, default=16.0)
  parser.add_argument("--backend_radius", type=int, default=2)
  parser.add_argument("--backend_nms", type=int, default=3)
  parser.add_argument("--disable_full_ba", action="store_true")

  parser.add_argument(
      "--mono_depth_path", default="Depth-Anything/video_visualization"
  )
  parser.add_argument("--metric_depth_path", default="UniDepth/outputs ")
  parser.add_argument(
      "--intrinsics",
      help="optional source-image camera intrinsics as a 3x3 K matrix or fx fy cx cy",
  )
  parser.add_argument(
      "--depth_source",
      "--depth-source",
      choices=("metric", "mono_aligned"),
      default="metric",
      help="metric feeds stereo/metric depth directly; mono_aligned keeps the original Depth-Anything alignment path",
  )
  parser.add_argument(
      "--valid_mask_path",
      "--valid-mask-path",
      help="optional directory of per-frame valid masks; white keeps pixels, black downweights/excludes them",
  )
  parser.add_argument(
      "--external_mask_weight",
      "--external-mask-weight",
      type=float,
      default=0.0,
      help="weight multiplier for black pixels in --valid_mask_path; 0.0 excludes them, 1.0 disables the external-mask effect",
  )
  parser.add_argument(
      "--mask_vis_path",
      "--mask-vis-path",
      help="optional output directory for visualizing the valid mask passed to Droid.track",
  )
  parser.add_argument(
      "--motion_mask_vis_path",
      "--motion-mask-vis-path",
      help="optional output directory for visualizing Mega-SAM's learned motion mask after full BA",
  )
  args = parser.parse_args()

  print("Running evaluation on {}".format(args.datapath))
  print(args)

  scene_name = args.scene_name.split("/")[-1]

  tstamps = []
  rgb_list = []
  senor_depth_list = []

  image_list = sorted(glob.glob(os.path.join("%s" % (args.datapath), "*.jpg")))
  image_list += sorted(glob.glob(os.path.join("%s" % (args.datapath), "*.png")))
  valid_mask_paths = resolve_mask_paths(args.valid_mask_path, image_list)
  mask_vis_dirs = prepare_mask_vis_dirs(args.mask_vis_path)
  if args.mask_vis_path:
    print(f"Saving valid mask visualization to {args.mask_vis_path}")
  if args.motion_mask_vis_path:
    print(f"Saving motion mask visualization to {args.motion_mask_vis_path}")

  # NOTE Mono is inverse depth, but metric-depth is depth!
  mono_disp_paths = sorted(
      glob.glob(
          os.path.join("%s/%s" % (args.mono_depth_path, scene_name), "*.npy")
      )
  )
  metric_depth_paths = sorted(
      glob.glob(
          os.path.join("%s/%s" % (args.metric_depth_path, scene_name), "*.npz")
      )
  )
  img_0 = cv2.imread(image_list[0])
  if img_0 is None:
    raise FileNotFoundError(f"Failed to read first image: {image_list[0]}")

  fovs = []
  aligns = None
  if args.depth_source == "metric":
    depth_list = []
    for metric_depth_file in metric_depth_paths:
      metric_depth, fov = load_metric_depth(metric_depth_file)
      depth_list.append(metric_depth)
      if np.isfinite(fov):
        fovs.append(fov)
  else:
    scales = []
    shifts = []
    depth_list = []
    for t, (mono_disp_file, metric_depth_file) in enumerate(
        zip(mono_disp_paths, metric_depth_paths)
    ):
      da_disp = np.float32(np.load(mono_disp_file))  # / 300.0
      uni_data = np.load(metric_depth_file)
      metric_depth = uni_data["depth"]

      fovs.append(uni_data["fov"])

      da_disp = cv2.resize(
          da_disp,
          (metric_depth.shape[1], metric_depth.shape[0]),
          interpolation=cv2.INTER_NEAREST_EXACT,
      )
      depth_list.append(da_disp)
      gt_disp = 1.0 / (metric_depth + 1e-8)

      # avoid some bug from UniDepth
      valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
      gt_disp[valid_mask] = 1e-2

      # avoid cases sky dominate entire video
      sky_ratio = np.sum(da_disp < 0.01) / (da_disp.shape[0] * da_disp.shape[1])
      if sky_ratio > 0.5:
        non_sky_mask = da_disp > 0.01
        gt_disp_ms = (
            gt_disp[non_sky_mask] - np.median(gt_disp[non_sky_mask]) + 1e-8
        )
        da_disp_ms = (
            da_disp[non_sky_mask] - np.median(da_disp[non_sky_mask]) + 1e-8
        )
        scale = np.median(gt_disp_ms / da_disp_ms)
        shift = np.median(gt_disp[non_sky_mask] - scale * da_disp[non_sky_mask])
      else:
        gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
        da_disp_ms = da_disp - np.median(da_disp) + 1e-8
        scale = np.median(gt_disp_ms / da_disp_ms)
        shift = np.median(gt_disp - scale * da_disp)

      gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
      da_disp_ms = da_disp - np.median(da_disp) + 1e-8

      scale = np.median(gt_disp_ms / da_disp_ms)
      shift = np.median(gt_disp - scale * da_disp)

      scales.append(scale)
      shifts.append(shift)

    ss_product = np.array(scales) * np.array(shifts)
    med_idx = np.argmin(np.abs(ss_product - np.median(ss_product)))

    align_scale = scales[med_idx]  # np.median(np.array(scales))
    align_shift = shifts[med_idx]  # np.median(np.array(shifts))
    normalize_scale = (
        np.percentile((align_scale * np.array(depth_list) + align_shift), 98)
        / 2.0
    )
    aligns = (align_scale, align_shift, normalize_scale)

  fallback_fov = np.median(fovs) if fovs else 60.0
  K = load_intrinsics(args.intrinsics, img_0.shape, fallback_fov)

  stream_count = min(len(image_list), len(depth_list))
  if stream_count == 0:
    raise RuntimeError("No synchronized image/depth frames found")
  if len(image_list) != len(depth_list):
    print(
        "WARNING: image/depth count mismatch; "
        f"using first {stream_count} frames "
        f"({len(image_list)} images, {len(depth_list)} {args.depth_source} depth maps)"
    )

  droid = None
  final_packet = None
  for t, image, depth, intrinsics, mask in tqdm(
      image_stream(
          image_list[:stream_count],
          depth_list[:stream_count],
          scene_name,
          use_depth=True,
          aligns=aligns,
          K=K,
          depth_source=args.depth_source,
          valid_mask_paths=valid_mask_paths,
      ),
      total=stream_count,
  ):
    save_mask_visualization(mask_vis_dirs, t, image, mask)
    if t == stream_count - 1:
      final_packet = (t, image, depth, intrinsics, mask)
      break

    if not args.disable_vis:
      show_image(image[0])

    rgb_list.append(image[0])
    senor_depth_list.append(depth)
    # breakpoint()
    if droid is None:
      args.image_size = [image.shape[2], image.shape[3]]
      droid = Droid(args)

    droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)

  if final_packet is None:
    raise RuntimeError("Failed to hold back final frame for track_final")
  t, image, depth, intrinsics, mask = final_packet
  if not args.disable_vis:
    show_image(image[0])
  rgb_list.append(image[0])
  senor_depth_list.append(depth)
  if droid is None:
    args.image_size = [image.shape[2], image.shape[3]]
    droid = Droid(args)

  # Process the final frame exactly once. Passing it through both track() and
  # track_final() can create duplicate/unstable poses at the end of a sequence.
  droid.track_final(t, image, depth, intrinsics=intrinsics, mask=mask)

  terminate_result = droid.terminate(
      image_stream(
          image_list[:stream_count],
          depth_list[:stream_count],
          scene_name,
          use_depth=True,
          aligns=aligns,
          K=K,
          depth_source=args.depth_source,
          valid_mask_paths=valid_mask_paths,
      ),
      _opt_intr=True,
      full_ba=not args.disable_full_ba,
      scene_name=scene_name,
  )

  if args.disable_full_ba:
    traj_est, depth_est = terminate_result
    motion_prob = np.zeros((traj_est.shape[0],), dtype=np.float32)
  else:
    traj_est, depth_est, motion_prob = terminate_result

  save_motion_mask_visualizations(args.motion_mask_vis_path, rgb_list, motion_prob)

  if args.scene_name is not None:
    save_full_reconstruction(
        droid,
        traj_est,
        rgb_list,
        senor_depth_list,
        motion_prob,
        args.scene_name,
    )
