#!/usr/bin/env python3
"""
Convert DNA 48-view data to CustomDUSt3R format.

Creates 48 separate "sequences" (one per camera view), treating each view's
temporal frames as a video sequence - exactly like DNA_02 format.

Output structure:
    output_dir/
    ├── {seq_name}_view0/
    │   ├── 00000.jpg  (frame 0 from view 0)
    │   ├── 00001.jpg  (frame 1 from view 0)
    │   └── ...
    ├── {seq_name}_view1/
    │   └── ...
    └── depth_maps/
        └── {seq_name}_view0/
            └── {seq_name}_viewmoge_calibrated.npy  (keyed by full image paths)

Usage:
    python convert_dna_to_custom_format.py --seq 0012_09 --output_dir ./data/DNA_48view
"""

import os
import sys
import json
import glob
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_dna_sequence(seq_name, data_dir, moge_dir, output_dir, use_symlinks=True, max_frames=None):
    """
    Convert a DNA sequence with 48 views to CustomDUSt3R format.
    Each view becomes a separate "sequence" for TTA.
    """
    
    seq_path = os.path.join(data_dir, seq_name)
    
    # Get all frames
    frames = sorted([d for d in os.listdir(seq_path) 
                    if d.isdigit() and os.path.isdir(os.path.join(seq_path, d))], key=int)
    
    if max_frames:
        frames = frames[:max_frames]
    
    print(f"Found {len(frames)} frames in {seq_name}")
    
    # Process each view
    for view_idx in tqdm(range(48), desc="Processing views"):
        view_seq_name = f"{seq_name}_view{view_idx}"
        
        # Create image folder
        view_images_dir = os.path.join(output_dir, view_seq_name)
        os.makedirs(view_images_dir, exist_ok=True)
        
        # Create depth folder
        view_depth_dir = os.path.join(output_dir, "depth_maps", view_seq_name)
        os.makedirs(view_depth_dir, exist_ok=True)
        
        # Load calibrated MoGe depth for this view
        moge_path = os.path.join(moge_dir, seq_name, f"{seq_name}_view{view_idx}", "moge_calibrated.npy")
        if not os.path.exists(moge_path):
            print(f"Warning: MoGe depth not found for view {view_idx}")
            continue
        
        try:
            moge_data = np.load(moge_path, allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading {moge_path}: {e}")
            continue
        
        # Load camera intrinsics from first frame
        cameras_path = os.path.join(seq_path, frames[0], "cameras.json")
        intrinsics_3x3 = None
        if os.path.exists(cameras_path):
            with open(cameras_path) as f:
                cameras = json.load(f)
            for cam in cameras:
                if int(cam['img_name']) == view_idx:
                    # Normalized intrinsics (0-1 range) - matches MoGe output format
                    intrinsics_3x3 = np.array([
                        [cam['fx'] / cam['width'], 0, 0.5],
                        [0, cam['fy'] / cam['height'], 0.5],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    break
        
        # Create depth dict keyed by FULL image paths
        depth_dict = {}
        
        for frame_idx, frame_name in enumerate(frames):
            # Source image path
            src_img_path = os.path.join(seq_path, frame_name, "rgbs", f"{view_idx:04d}.png")
            
            if not os.path.exists(src_img_path):
                continue
            
            # Destination image path - format: 00000.jpg, 00001.jpg, ...
            dst_img_name = f"{frame_idx:05d}.png"
            dst_img_path = os.path.join(view_images_dir, dst_img_name)
            
            # Process image: center crop to 1024x1024, then resize to 512x512
            if not os.path.exists(dst_img_path):
                img = Image.open(src_img_path)
                w, h = img.size  # 1024 x 1224 typically
                
                # Center crop to 1024x1024 (square)
                crop_size = min(w, h)  # 1024
                if crop_size != 1024:
                    crop_size = 1024  # Force 1024
                left = (w - crop_size) // 2
                top = (h - crop_size) // 2
                right = left + crop_size
                bottom = top + crop_size
                img_cropped = img.crop((left, top, right, bottom))
                
                # Resize to 512x512
                img_resized = img_cropped.resize((512, 512), Image.LANCZOS)
                img_resized.save(dst_img_path)
            
            # Get depth data for this frame
            frame_key = f"frame_{int(frame_name):04d}"
            if frame_key in moge_data:
                depth_entry = moge_data[frame_key]
                depth = depth_entry['depth'].copy().astype(np.float32)
                mask = depth_entry.get('mask', np.ones_like(depth, dtype=np.float32))
                if mask.dtype == bool:
                    mask = mask.astype(np.float32)
                
                # Handle NaN/Inf
                nan_inf_mask = np.isnan(depth) | np.isinf(depth)
                depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                mask = mask * (~nan_inf_mask).astype(np.float32)
                
                # Get points
                points = depth_entry.get('points', None)
                if points is None:
                    H, W = depth.shape
                    points = np.zeros((H, W, 3), dtype=np.float32)
                else:
                    points = points.copy().astype(np.float32)
                    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Scale factor (if available)
                scale_factor = depth_entry.get('scale_factor', 1.0)
                
                # Key is the FULL path to the destination image
                depth_dict[dst_img_path] = {
                    'depth': depth,
                    'mask': mask,
                    'intrinsics': intrinsics_3x3 if intrinsics_3x3 is not None else np.eye(3, dtype=np.float32),
                    'points': points,
                    'scale_factor': float(scale_factor)
                }
        
        # Save depth dict with the naming convention CustomDUSt3R expects
        # It looks for: depth_path + seq.split('/')[-2] + depth_filename
        # So if depth_path='./depth_maps/{view}/' and seq='./data/{view}/'
        # It will look for './depth_maps/{view}/{view}moge_calibrated.npy'
        depth_out_path = os.path.join(view_depth_dir, f"{view_seq_name}moge_calibrated.npy")
        np.save(depth_out_path, depth_dict)
        
        if view_idx % 10 == 0:
            print(f"  View {view_idx}: {len(depth_dict)} frames saved")
    
    print(f"\n" + "="*60)
    print(f"Done! Output saved to {output_dir}")
    print(f"\nTo use with CustomDUSt3R TTA:")
    print(f"  dataset_location='{output_dir}/{seq_name}_view*/'")
    print(f"  depth_path='{output_dir}/depth_maps/'")
    print(f"  depth_filename='moge_calibrated.npy'")
    print(f"="*60)


def main():
    parser = argparse.ArgumentParser(description="Convert DNA 48-view data to CustomDUSt3R format")
    parser.add_argument("--seq", default="0012_09", help="Sequence name")
    parser.add_argument("--data_dir", default="/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/Part1",
                        help="Path to DNA Part1 folder")
    parser.add_argument("--moge_dir", default="/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/calibrated_depth",
                        help="Path to calibrated_depth folder")
    parser.add_argument("--output_dir", default="./data/DNA_48view",
                        help="Output directory")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinks")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames per view")
    
    args = parser.parse_args()
    
    convert_dna_sequence(
        args.seq,
        args.data_dir,
        args.moge_dir,
        args.output_dir,
        use_symlinks=not args.copy,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
