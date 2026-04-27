import h5py
import cv2
import numpy as np
import os

def extract_egodex_sequence(hdf5_path, mp4_path, output_dir):
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    # 读取HDF5元数据
    with h5py.File(hdf5_path, 'r') as f:
        # 相机内参（固定）
        K = f['camera/intrinsic'][:]
        np.savetxt(f"{output_dir}/camera_intrinsics.txt", K)
        
        # 相机位姿真值（相机到 ARKit 原点/世界坐标的变换矩阵）
        gt_poses = f['transforms/camera'][:]  # N x 4 x 4
        np.savetxt(f"{output_dir}/ground_truth_poses.txt", gt_poses.reshape(-1, 16))
    
    # 提取视频帧
    cap = cv2.VideoCapture(mp4_path)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_dir}/images/{frame_id:06d}.png", frame)
        frame_id += 1
    cap.release()
    
    print(f"提取完成：共{frame_id}帧，保存到{output_dir}")

# 使用示例
extract_egodex_sequence(
    hdf5_path="/home/user/test/add_remove_lid/0.hdf5",
    mp4_path="/home/user/test/add_remove_lid/0.mp4",
    output_dir="./egodex_seq_0"
)
