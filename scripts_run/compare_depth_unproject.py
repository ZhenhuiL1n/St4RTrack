#!/usr/bin/env python3
"""
Compare unprojected point clouds from:
1. GT rendered depth (from PCL)
2. MoGe calibrated depth

This helps visualize if they align in 3D space.
"""

import os
import numpy as np
import json
import cv2
import open3d as o3d

# Configuration
BASE_DATA_DIR = "/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/Part1"
CALIBRATED_DIR = "/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/calibrated_depth"

# Select sequence, frame, view
SEQ_NAME = "0012_09"
FRAME_IDX = 50
VIEW_IDX = 22


def centre_crop(img, size=1024):
    h, w = img.shape[:2]
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    return img[start_h:start_h + size, start_w:start_w + size]


def load_camera(seq_dir, frame_idx, view_idx):
    """Load camera parameters."""
    cameras_path = os.path.join(seq_dir, str(frame_idx), "cameras.json")
    with open(cameras_path, 'r') as f:
        cameras = json.load(f)
    
    for cam in cameras:
        if cam['img_name'] == f"{view_idx:04d}":
            return cam
    return None


def unproject_depth(depth, K, mask=None):
    """Unproject depth to 3D points in camera coordinates."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    if mask is None:
        mask = depth > 0
    
    valid = mask & (depth > 0)
    
    z = depth[valid]
    x = (u[valid] - K[0, 2]) * z / K[0, 0]
    y = (v[valid] - K[1, 2]) * z / K[1, 1]
    
    points = np.stack([x, y, z], axis=-1)
    return points, valid


def main():
    seq_dir = os.path.join(BASE_DATA_DIR, SEQ_NAME)
    
    # Load camera
    cam = load_camera(seq_dir, FRAME_IDX, VIEW_IDX)
    if cam is None:
        print(f"Camera not found for view {VIEW_IDX}")
        return
    
    # Build intrinsic matrix for 512x512 (after center crop + resize)
    # Original: 1024x1224, center crop to 1024x1024, resize to 512x512
    scale = 512.0 / 1024.0
    fx = cam['fx'] * scale
    fy = cam['fy'] * scale
    cx = 256.0  # Center of 512
    cy = 256.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    print(f"Intrinsic matrix K (for 512x512):\n{K}")
    
    # ==========================================
    # 1. Load GT rendered depth (from PCL)
    # ==========================================
    gt_depth_path = os.path.join(seq_dir, str(FRAME_IDX), "depth_rendered", f"{VIEW_IDX:04d}.npy")
    gt_depth_raw = np.load(gt_depth_path)  # 1024x1224
    print(f"\nGT depth raw shape: {gt_depth_raw.shape}")
    
    # Center crop to 1024x1024, resize to 512x512
    gt_depth = centre_crop(gt_depth_raw, 1024)
    gt_depth = cv2.resize(gt_depth, (512, 512), interpolation=cv2.INTER_NEAREST)
    print(f"GT depth after crop+resize: {gt_depth.shape}")
    print(f"GT depth range: {gt_depth[gt_depth>0].min():.3f} - {gt_depth[gt_depth>0].max():.3f}")
    
    # ==========================================
    # 2. Load MoGe calibrated depth
    # ==========================================
    calib_dir = os.path.join(CALIBRATED_DIR, SEQ_NAME, f"{SEQ_NAME}_view{VIEW_IDX}")
    calib_path = os.path.join(calib_dir, "moge_calibrated.npy")
    
    if os.path.exists(calib_path):
        moge_data = np.load(calib_path, allow_pickle=True).item()
        frame_key = f"frame_{FRAME_IDX:04d}"
        moge_depth = moge_data[frame_key]['depth']  # Already 512x512
        print(f"\nMoGe calibrated depth shape: {moge_depth.shape}")
        print(f"MoGe depth range: {moge_depth[moge_depth>0].min():.3f} - {moge_depth[moge_depth>0].max():.3f}")
    else:
        print(f"\nMoGe calibrated depth not found at: {calib_path}")
        moge_depth = None
    
    # ==========================================
    # 3. Load foreground mask
    # ==========================================
    mask_path = os.path.join(seq_dir, str(FRAME_IDX), "masks", f"{VIEW_IDX:04d}.png")
    if os.path.exists(mask_path):
        fg_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        fg_mask = centre_crop(fg_mask, 1024)
        fg_mask = cv2.resize(fg_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        fg_mask = fg_mask > 127
        print(f"\nForeground mask: {fg_mask.sum()} pixels")
    else:
        fg_mask = gt_depth > 0
        print("\nUsing depth > 0 as mask")
    
    # ==========================================
    # 4. Unproject both to 3D
    # ==========================================
    print("\n" + "="*60)
    print("Unprojecting to 3D (foreground only)...")
    
    gt_pts, gt_valid = unproject_depth(gt_depth, K, fg_mask)
    print(f"GT points: {len(gt_pts)}")
    
    if moge_depth is not None:
        moge_pts, moge_valid = unproject_depth(moge_depth, K, fg_mask)
        print(f"MoGe points: {len(moge_pts)}")
        
        # ==========================================
        # 5. Compare statistics
        # ==========================================
        print("\n" + "="*60)
        print("COMPARISON (foreground):")
        
        # Depth values comparison
        gt_fg_depth = gt_depth[fg_mask & (gt_depth > 0)]
        moge_fg_depth = moge_depth[fg_mask & (moge_depth > 0)]
        
        print(f"\nGT depth     - mean: {gt_fg_depth.mean():.3f}, std: {gt_fg_depth.std():.3f}")
        print(f"MoGe depth   - mean: {moge_fg_depth.mean():.3f}, std: {moge_fg_depth.std():.3f}")
        
        # Compute difference
        valid_both = fg_mask & (gt_depth > 0) & (moge_depth > 0)
        diff = np.abs(gt_depth[valid_both] - moge_depth[valid_both])
        print(f"\nPixel-wise difference:")
        print(f"  Mean: {diff.mean():.4f}")
        print(f"  Max:  {diff.max():.4f}")
        print(f"  Median: {np.median(diff):.4f}")
        
        # Relative error
        rel_err = diff / gt_depth[valid_both]
        print(f"\nRelative error:")
        print(f"  Mean: {rel_err.mean()*100:.2f}%")
        print(f"  Median: {np.median(rel_err)*100:.2f}%")
    
    # ==========================================
    # 6. Save point clouds for visualization
    # ==========================================
    print("\n" + "="*60)
    print("Saving point clouds...")
    
    # GT points (red)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
    pcd_gt.paint_uniform_color([1, 0, 0])  # Red
    o3d.io.write_point_cloud("compare_gt_depth.ply", pcd_gt)
    print(f"  Saved: compare_gt_depth.ply (RED)")
    
    if moge_depth is not None:
        # MoGe points (blue)
        pcd_moge = o3d.geometry.PointCloud()
        pcd_moge.points = o3d.utility.Vector3dVector(moge_pts)
        pcd_moge.paint_uniform_color([0, 0, 1])  # Blue
        o3d.io.write_point_cloud("compare_moge_depth.ply", pcd_moge)
        print(f"  Saved: compare_moge_depth.ply (BLUE)")
        
        # Combined
        pcd_combined = pcd_gt + pcd_moge
        o3d.io.write_point_cloud("compare_combined.ply", pcd_combined)
        print(f"  Saved: compare_combined.ply (RED=GT, BLUE=MoGe)")
    
    print("\nDone! Open compare_combined.ply in MeshLab to visualize.")


if __name__ == "__main__":
    main()
