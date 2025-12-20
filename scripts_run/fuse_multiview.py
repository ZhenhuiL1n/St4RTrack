#!/usr/bin/env python3
"""
Fuse calibrated depth from all 48 views into a complete human body point cloud.

This script:
1. Loads calibrated depth from all 48 views for a specific frame
2. Applies foreground mask to extract only the human
3. Unprojects each view to 3D using camera parameters
4. Transforms all views to world coordinates
5. Fuses into a single colored point cloud

Usage:
    python fuse_multiview.py --seq 0012_09 --frame 50
    python fuse_multiview.py --seq 0012_09 --frame 50 --output fused_human.ply
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import open3d as o3d

# Configuration
BASE_DATA_DIR = "/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/Part1"
CALIBRATED_DIR = "/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/calibrated_depth"
NUM_VIEWS = 48


def centre_crop(img, size=1024):
    """Center crop image to square."""
    h, w = img.shape[:2]
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    return img[start_h:start_h + size, start_w:start_w + size]


def get_camera_for_view(cameras_json_path, view_idx):
    """Get camera parameters for a specific view."""
    with open(cameras_json_path, 'r') as f:
        cameras = json.load(f)
    
    for cam in cameras:
        if cam['img_name'] == f"{view_idx:04d}":
            return cam
    return None


def unproject_depth_to_3d(depth_map, intrinsics, mask=None):
    """Unproject depth map to 3D points in camera coordinates."""
    h, w = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    if mask is None:
        valid = depth_map > 0
    else:
        valid = mask & (depth_map > 0)
    
    z = depth_map[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    
    pts_3d = np.stack([x, y, z], axis=-1)
    return pts_3d, valid


def transform_points_to_world(points_cam, cam_params):
    """Transform points from camera coordinates to world coordinates.
    
    Camera stores: position (C) and rotation (R_wc = world-to-camera rotation)
    World-to-camera: P_cam = R_cw @ P_world + t_cw
                   where R_cw = R_wc.T, t_cw = -R_cw @ C
    
    Camera-to-world (inverse): P_world = R_cw.T @ (P_cam - t_cw)
                                       = R_wc @ (P_cam - t_cw)
                                       = (P_cam - t_cw) @ R_wc.T  (for batch)
                                       = (P_cam - t_cw) @ R_cw
    """
    pos = np.array(cam_params['position'])
    rot = np.array(cam_params['rotation'])
    
    # R_wc is stored in cameras.json
    R_wc = rot
    R_cw = R_wc.T
    t_cw = -R_cw @ pos
    
    # Transform from camera to world: P_world = (P_cam - t_cw) @ R_cw
    points_world = (points_cam - t_cw) @ R_cw
    
    return points_world


def fuse_views(seq_name, frame_idx, output_path=None, use_color=True):
    """Fuse all 48 views into a single point cloud."""
    
    print(f"\n{'='*60}")
    print(f"Fusing {NUM_VIEWS} views for {seq_name} frame {frame_idx}")
    print(f"{'='*60}")
    
    seq_dir = os.path.join(BASE_DATA_DIR, seq_name)
    calib_dir = os.path.join(CALIBRATED_DIR, seq_name)
    
    # Check if calibrated data exists
    if not os.path.exists(calib_dir):
        print(f"✗ Error: Calibrated data not found at {calib_dir}")
        return None
    
    # Get camera parameters from first frame
    cameras_path = os.path.join(seq_dir, str(frame_idx), "cameras.json")
    if not os.path.exists(cameras_path):
        print(f"✗ Error: cameras.json not found at {cameras_path}")
        return None
    
    all_points = []
    all_colors = []
    
    for view_idx in tqdm(range(NUM_VIEWS), desc="Processing views"):
        # Load calibrated depth
        view_dir = os.path.join(calib_dir, f"{seq_name}_view{view_idx}")
        calib_path = os.path.join(view_dir, "moge_calibrated.npy")
        
        if not os.path.exists(calib_path):
            continue
        
        calib_data = np.load(calib_path, allow_pickle=True).item()
        frame_key = f"frame_{frame_idx:04d}"
        
        if frame_key not in calib_data:
            continue
        
        frame_data = calib_data[frame_key]
        depth = frame_data['depth']
        mask = frame_data['mask'].astype(bool)
        intrinsics = frame_data['intrinsics']
        
        # Load foreground mask
        mask_path = os.path.join(seq_dir, str(frame_idx), "masks", f"{view_idx:04d}.png")
        if os.path.exists(mask_path):
            fg_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            fg_mask = centre_crop(fg_mask, 1024)
            fg_mask = cv2.resize(fg_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            fg_mask = fg_mask > 127
        else:
            fg_mask = mask
        
        # Get camera parameters
        cam = get_camera_for_view(cameras_path, view_idx)
        if cam is None:
            continue
        
        # Use ACTUAL camera intrinsics (from cameras.json)
        # Original image is 1024x1224, we center-cropped to 1024x1024 and resized to 512x512
        # So scale factor is 512/1024 = 0.5
        scale = 512.0 / 1024.0
        fx = cam['fx'] * scale
        fy = cam['fy'] * scale
        # Original cx was at width/2, after center crop it stays at 512 (half of 1024), scaled -> 256
        cx = 256.0
        cy = 256.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        # Unproject foreground to 3D (camera coordinates)
        pts_cam, valid_mask = unproject_depth_to_3d(depth, K, fg_mask)
        
        if len(pts_cam) == 0:
            continue
        
        # Transform to world coordinates
        pts_world = transform_points_to_world(pts_cam, cam)
        all_points.append(pts_world)
        
        # Get colors from RGB image
        if use_color:
            rgb_path = os.path.join(seq_dir, str(frame_idx), "rgbs", f"{view_idx:04d}.png")
            if os.path.exists(rgb_path):
                rgb = cv2.imread(rgb_path)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = centre_crop(rgb, 1024)
                rgb = cv2.resize(rgb, (512, 512))
                colors = rgb[valid_mask] / 255.0
                all_colors.append(colors)
            else:
                # Random color per view for debugging
                color = np.random.rand(3)
                all_colors.append(np.tile(color, (len(pts_world), 1)))
        else:
            # Random color per view
            color = np.random.rand(3)
            all_colors.append(np.tile(color, (len(pts_world), 1)))
    
    if len(all_points) == 0:
        print("✗ Error: No valid views found")
        return None
    
    # Combine all points
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    
    print(f"\n  Total points: {len(all_points):,}")
    print(f"  Views processed: {len([p for p in all_points])}")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # Optional: Remove outliers
    print("  Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  Points after filtering: {len(pcd.points):,}")
    
    # Save
    if output_path is None:
        output_path = f"fused_{seq_name}_frame{frame_idx:04d}.ply"
    
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"\n  ✓ Saved to: {output_path}")
    
    return pcd


def main():
    parser = argparse.ArgumentParser(description="Fuse multi-view calibrated depth into human point cloud")
    parser.add_argument('--seq', type=str, required=True, help='Sequence name (e.g., 0012_09)')
    parser.add_argument('--frame', type=int, required=True, help='Frame index (e.g., 50)')
    parser.add_argument('--output', type=str, default=None, help='Output PLY path')
    parser.add_argument('--no_color', action='store_true', help='Use random colors instead of RGB')
    args = parser.parse_args()
    
    pcd = fuse_views(
        args.seq,
        args.frame,
        output_path=args.output,
        use_color=not args.no_color
    )
    
    if pcd is not None:
        print("\n✓ Fusion complete! Open the PLY file in MeshLab or CloudCompare to visualize.")


if __name__ == "__main__":
    main()
