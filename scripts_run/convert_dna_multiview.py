#!/usr/bin/env python3
"""
Convert DNA 48-view data to CustomDUSt3R multi-view format.

Treats each TIMESTAMP as a "sequence" and each VIEW as a "frame".
This way, CustomDUSt3R with S=8 will sample 8 different views from the same timestamp.

Output structure:
    output_dir/
    ├── {seq_name}_t0000/     (timestamp 0)
    │   ├── 00.png           (view 0)
    │   ├── 01.png           (view 1)
    │   └── ...              (48 views total)
    ├── {seq_name}_t0001/     (timestamp 1)
    │   └── ...
    └── depth_maps/
        └── {seq_name}_t0000/
            └── {seq_name}_t0000moge_calibrated.npy

Usage:
    python convert_dna_multiview.py --seq 0012_09
    
Then train with:
    dataset_location='./data/DNA_multiview/0012_09_t*/'
    depth_path='./data/DNA_multiview/depth_maps/'
    depth_filename='moge_calibrated.npy'
"""

import os
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm

def convert_multiview(seq_name, data_dir, moge_dir, output_dir, max_frames=None):
    """
    Convert DNA 48-view data to multi-view format.
    Each timestamp becomes a sequence, each view becomes a frame.
    """
    
    seq_path = os.path.join(data_dir, seq_name)
    
    # Get all timestamps (frames in DNA terminology)
    timestamps = sorted([d for d in os.listdir(seq_path) 
                        if d.isdigit() and os.path.isdir(os.path.join(seq_path, d))], key=int)
    
    if max_frames:
        timestamps = timestamps[:max_frames]
    
    print(f"Found {len(timestamps)} timestamps in {seq_name}")
    
    # Pre-load all MoGe calibrated depth data
    print("Loading MoGe depth data...")
    moge_cache = {}
    for view_idx in range(48):
        moge_path = os.path.join(moge_dir, seq_name, f"{seq_name}_view{view_idx}", "moge_calibrated.npy")
        if os.path.exists(moge_path):
            try:
                moge_cache[view_idx] = np.load(moge_path, allow_pickle=True).item()
            except Exception as e:
                print(f"Error loading view {view_idx}: {e}")
    print(f"Loaded {len(moge_cache)} view depth files")
    
    # Load camera intrinsics from first frame
    cameras_path = os.path.join(seq_path, timestamps[0], "cameras.json")
    intrinsics_dict = {}
    if os.path.exists(cameras_path):
        with open(cameras_path) as f:
            cameras = json.load(f)
        for cam in cameras:
            view_idx = int(cam['img_name'])
            # Normalized intrinsics (0-1 range)
            intrinsics_dict[view_idx] = np.array([
                [cam['fx'] / cam['width'], 0, 0.5],
                [0, cam['fy'] / cam['height'], 0.5],
                [0, 0, 1]
            ], dtype=np.float32)
    
    # Process each timestamp
    for timestamp in tqdm(timestamps, desc="Processing timestamps"):
        ts_name = f"{seq_name}_t{int(timestamp):04d}"
        
        # Create image folder
        ts_images_dir = os.path.join(output_dir, ts_name)
        os.makedirs(ts_images_dir, exist_ok=True)
        
        # Create depth folder
        ts_depth_dir = os.path.join(output_dir, "depth_maps", ts_name)
        os.makedirs(ts_depth_dir, exist_ok=True)
        
        # Create depth dict for this timestamp
        depth_dict = {}
        
        for view_idx in range(48):
            # Source image path
            src_img_path = os.path.join(seq_path, timestamp, "rgbs", f"{view_idx:04d}.png")
            
            if not os.path.exists(src_img_path):
                continue
            
            # Destination image path - format: 00.png, 01.png, ... (view index)
            dst_img_name = f"{view_idx:02d}.png"
            dst_img_path = os.path.join(ts_images_dir, dst_img_name)
            
            # Create symlink
            if not os.path.exists(dst_img_path):
                os.symlink(os.path.abspath(src_img_path), dst_img_path)
            
            # Get depth data
            frame_key = f"frame_{int(timestamp):04d}"
            if view_idx in moge_cache and frame_key in moge_cache[view_idx]:
                depth_entry = moge_cache[view_idx][frame_key]
                depth = depth_entry['depth'].copy().astype(np.float32)
                mask = depth_entry.get('mask', np.ones_like(depth, dtype=np.float32))
                if mask.dtype == bool:
                    mask = mask.astype(np.float32)
                
                # Handle NaN/Inf
                nan_inf_mask = np.isnan(depth) | np.isinf(depth)
                depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                mask = mask * (~nan_inf_mask).astype(np.float32)
                
                # Get points
                points = depth_entry.get('points', np.zeros((*depth.shape, 3), dtype=np.float32))
                if points is not None:
                    points = np.nan_to_num(points.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                
                scale_factor = depth_entry.get('scale_factor', 1.0)
                
                depth_dict[dst_img_path] = {
                    'depth': depth,
                    'mask': mask,
                    'intrinsics': intrinsics_dict.get(view_idx, np.eye(3, dtype=np.float32)),
                    'points': points,
                    'scale_factor': float(scale_factor)
                }
        
        # Save depth dict
        depth_out_path = os.path.join(ts_depth_dir, f"{ts_name}moge_calibrated.npy")
        np.save(depth_out_path, depth_dict)
    
    print(f"\n" + "="*70)
    print(f"Done! Output saved to {output_dir}")
    print(f"\nStructure:")
    print(f"  - {len(timestamps)} timestamp 'sequences'")
    print(f"  - Each with 48 view 'frames'")
    print(f"\nTo use with CustomDUSt3R TTA:")
    print(f"  dataset_location='{output_dir}/{seq_name}_t*/'")
    print(f"  depth_path='{output_dir}/depth_maps/'")
    print(f"  depth_filename='moge_calibrated.npy'")
    print(f"  S=8  # Sample 8 different views per batch")
    print(f"="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="0012_09")
    parser.add_argument("--data_dir", default="/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/Part1")
    parser.add_argument("--moge_dir", default="/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/calibrated_depth")
    parser.add_argument("--output_dir", default="./data/DNA_multiview")
    parser.add_argument("--max_frames", type=int, default=None)
    
    args = parser.parse_args()
    convert_multiview(args.seq, args.data_dir, args.moge_dir, args.output_dir, args.max_frames)


if __name__ == "__main__":
    main()
