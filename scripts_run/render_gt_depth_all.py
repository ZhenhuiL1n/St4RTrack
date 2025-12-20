#!/usr/bin/env python3
"""
Re-render GT depth from point clouds for all sequences/frames/views.
Replaces the existing depth_rendered/{view}.npy files.

This uses the UNCENTERED point cloud and correct camera extrinsics.

Usage:
    # All sequences in parallel (7 GPUs, but no GPU needed for this)
    python render_gt_depth_all.py --workers 16
    
    # Single sequence
    python render_gt_depth_all.py --seq 0012_09 --workers 8
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import open3d as o3d
from multiprocessing import Pool, cpu_count

# Configuration
BASE_DATA_DIR = "/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/Part1"
NUM_VIEWS = 48


def find_best_ply(frame_dir):
    """Find the best available PLY file (highest iteration)."""
    pcl_dir = os.path.join(frame_dir, "point_cloud")
    if not os.path.exists(pcl_dir):
        return None
    
    iterations = []
    for item in os.listdir(pcl_dir):
        if item.startswith("iteration_"):
            try:
                iter_num = int(item.split("_")[1])
                ply_path = os.path.join(pcl_dir, item, "point_cloud.ply")
                if os.path.exists(ply_path):
                    iterations.append((iter_num, ply_path))
            except:
                pass
    
    if not iterations:
        return None
    
    iterations.sort(reverse=True)
    return iterations[0][1]


def render_depth_for_frame(args):
    """Render GT depth for one frame, all views."""
    seq_name, frame_idx = args
    
    frame_dir = os.path.join(BASE_DATA_DIR, seq_name, str(frame_idx))
    
    if not os.path.exists(frame_dir):
        return (seq_name, frame_idx, False, "Frame dir not found")
    
    # Find best PLY (any iteration)
    ply_path = find_best_ply(frame_dir)
    if not ply_path:
        return (seq_name, frame_idx, False, "PLY not found")
    
    # Load cameras
    cameras_path = os.path.join(frame_dir, "cameras.json")
    if not os.path.exists(cameras_path):
        return (seq_name, frame_idx, False, "cameras.json not found")
    
    with open(cameras_path, 'r') as f:
        cameras = json.load(f)
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    points_world = np.asarray(pcd.points)
    
    if len(points_world) == 0:
        return (seq_name, frame_idx, False, "Empty point cloud")
    
    # Create output directory
    output_dir = os.path.join(frame_dir, "depth_rendered")
    os.makedirs(output_dir, exist_ok=True)
    
    views_rendered = 0
    
    for view_idx in range(NUM_VIEWS):
        # Find camera for this view
        cam = None
        for c in cameras:
            if c['img_name'] == f"{view_idx:04d}":
                cam = c
                break
        
        if cam is None:
            continue
        
        # Get camera parameters
        width, height = cam['width'], cam['height']
        fx, fy = cam['fx'], cam['fy']
        cx, cy = width / 2.0, height / 2.0
        
        pos = np.array(cam['position'])
        rot = np.array(cam['rotation'])
        
        # World to camera transform
        R_cw = rot.T
        t_cw = -R_cw @ pos
        
        # Transform points to camera frame
        points_cam = (points_world @ R_cw.T) + t_cw
        
        # Filter points in front of camera
        valid_z = points_cam[:, 2] > 0.01
        points_cam_valid = points_cam[valid_z]
        
        if len(points_cam_valid) == 0:
            continue
        
        # Project to image
        z = points_cam_valid[:, 2]
        u = (points_cam_valid[:, 0] * fx / z) + cx
        v = (points_cam_valid[:, 1] * fy / z) + cy
        
        # Filter inside image bounds
        valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u = u[valid_uv]
        v = v[valid_uv]
        z = z[valid_uv]
        
        if len(z) == 0:
            continue
        
        # Create depth map (splat points, further points first)
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        sort_idx = np.argsort(z)[::-1]  # Far to near
        u_sorted = np.round(u[sort_idx]).astype(int)
        v_sorted = np.round(v[sort_idx]).astype(int)
        z_sorted = z[sort_idx]
        
        # Splat with small radius for better coverage
        for i in range(len(z_sorted)):
            cv2.circle(depth_map, (u_sorted[i], v_sorted[i]), 2, float(z_sorted[i]), -1)
        
        # Save
        output_path = os.path.join(output_dir, f"{view_idx:04d}.npy")
        np.save(output_path, depth_map)
        views_rendered += 1
    
    return (seq_name, frame_idx, True, f"Rendered {views_rendered} views")


def get_all_frames(seq_name):
    """Get all frame indices for a sequence."""
    seq_dir = os.path.join(BASE_DATA_DIR, seq_name)
    if not os.path.exists(seq_dir):
        return []
    
    frames = []
    for item in os.listdir(seq_dir):
        if item.isdigit():
            frames.append(int(item))
    return sorted(frames)


def get_all_sequences():
    """Get all sequence names."""
    sequences = []
    for item in os.listdir(BASE_DATA_DIR):
        if os.path.isdir(os.path.join(BASE_DATA_DIR, item)):
            sequences.append(item)
    return sorted(sequences)


def main():
    parser = argparse.ArgumentParser(description="Render GT depth for all sequences/frames/views")
    parser.add_argument('--seq', type=str, default=None, help='Single sequence to process')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel workers')
    args = parser.parse_args()
    
    # Get sequences to process
    if args.seq:
        sequences = [args.seq]
    else:
        sequences = get_all_sequences()
    
    print(f"{'='*60}")
    print(f"Rendering GT depth for {len(sequences)} sequences")
    print(f"Workers: {args.workers}")
    print(f"{'='*60}")
    
    # Build work list
    work_items = []
    for seq_name in sequences:
        frames = get_all_frames(seq_name)
        for frame_idx in frames:
            work_items.append((seq_name, frame_idx))
    
    print(f"Total frames to process: {len(work_items)}")
    
    # Process in parallel
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(render_depth_for_frame, work_items),
            total=len(work_items),
            desc="Rendering GT depth"
        ))
    
    # Report
    success = sum(1 for r in results if r[2])
    failed = [(r[0], r[1], r[3]) for r in results if not r[2]]
    
    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"{'='*60}")
    print(f"  Successful: {success}/{len(work_items)}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for seq, frame, msg in failed[:10]:
            print(f"    {seq}/{frame}: {msg}")


if __name__ == "__main__":
    main()
