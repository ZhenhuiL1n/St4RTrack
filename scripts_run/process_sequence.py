#!/usr/bin/env python3
"""
Process DNA-Rendering sequences - Complete Depth Calibration Pipeline
Supports multiple sequences and multiple views.

Step-by-step workflow:
1. Extract RGB frames to reproduction/{seq}_view{view}/1_frames_raw/
2. Center crop and resize to 512x512 -> 2_frames_512/
3. Run MoGe depth estimation -> 3_moge_depth/
   3b. (Alternative) Load existing MoGe data
4. Render PCL depth at raw resolution (1024x1224) -> 4_rendered_depth_raw/
5. Center crop and resize rendered depth -> 5_rendered_depth_512/
6. Calibrate MoGe using rendered depth -> 6_calibrated_depth/
7. Create visualizations -> 7_visualizations/
8. Create unprojected PLY comparisons -> 8_unprojected_ply/
9. Create combined calibrated npy file -> moge_calibrated.npy

Usage:
    # Single view
    python process_sequence.py --seq 0012_09 --view 22 --step all
    
    # Multiple views
    python process_sequence.py --seq 0012_09 --views 0,10,22,30 --step all
    
    # All 48 views
    python process_sequence.py --seq 0012_09 --views all --step all
    
    # Specific steps
    python process_sequence.py --seq 0012_09 --view 22 --step 1,2,3,4,5,6,7,8,9
    
    # Load existing MoGe instead of running inference
    python process_sequence.py --seq 0012_09 --view 22 --step all --load_moge
    
    # FAST mode: skip intermediate files, only produce moge_calibrated.npy
    python process_sequence.py --seq 0012_09 --views all --fast --load_moge
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import open3d as o3d

# Add paths
sys.path.append('/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack')
sys.path.append('/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/third_party/MoGe')

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
NUM_FRAMES = 150
NUM_VIEWS = 48

# Source data
BASE_DATA_DIR = "/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/Part1"

# Existing MoGe data path pattern
EXISTING_MOGE_PATTERN = "/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/data/DNA_Seq/depth_maps/{seq}/{seq}moge_results.npy"

# Output base directory
OUTPUT_BASE = "/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/reproduction"


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def centre_crop(img, size=1024):
    """Center crop image to square."""
    h, w = img.shape[:2]
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    return img[start_h:start_h + size, start_w:start_w + size]


def depth_to_color(depth, vmin=None, vmax=None, colormap=cv2.COLORMAP_VIRIDIS):
    """Convert depth map to colored visualization."""
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    
    if vmin is None:
        vmin = valid.min()
    if vmax is None:
        vmax = valid.max()
    
    norm = (depth - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0, 1)
    norm[depth <= 0] = 0.0
    return cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)


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


def create_colored_pointcloud(pts_3d, color):
    """Create Open3D point cloud with uniform color."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts_3d), 1)))
    return pcd


# ======================================================================
# Pipeline class
# ======================================================================
class DepthCalibrationPipeline:
    def __init__(self, seq_name, view_idx, num_frames=150, fast_mode=False, output_base=None):
        self.seq_name = seq_name
        self.view_idx = view_idx
        self.num_frames = num_frames
        self.fast_mode = fast_mode  # Skip saving intermediate files
        self.output_base = output_base or OUTPUT_BASE
        
        # Source data
        self.seq_dir = os.path.join(BASE_DATA_DIR, seq_name)
        
        # Output directory
        self.repro_dir = os.path.join(self.output_base, f"{seq_name}_view{view_idx}")
        
        # Subfolder structure
        self.frames_raw_dir = os.path.join(self.repro_dir, "1_frames_raw")
        self.frames_512_dir = os.path.join(self.repro_dir, "2_frames_512")
        self.masks_raw_dir = os.path.join(self.repro_dir, "1_masks_raw")
        self.masks_512_dir = os.path.join(self.repro_dir, "2_masks_512")
        self.moge_depth_dir = os.path.join(self.repro_dir, "3_moge_depth")
        self.moge_vis_dir = os.path.join(self.repro_dir, "3_moge_vis")
        self.rendered_raw_dir = os.path.join(self.repro_dir, "4_rendered_depth_raw")
        self.rendered_512_dir = os.path.join(self.repro_dir, "5_rendered_depth_512")
        self.calibrated_dir = os.path.join(self.repro_dir, "6_calibrated_depth")
        self.vis_dir = os.path.join(self.repro_dir, "7_visualizations")
        self.ply_dir = os.path.join(self.repro_dir, "8_unprojected_ply")
        
        self.all_dirs = [
            self.repro_dir, self.frames_raw_dir, self.frames_512_dir,
            self.masks_raw_dir, self.masks_512_dir, self.moge_depth_dir,
            self.moge_vis_dir, self.rendered_raw_dir, self.rendered_512_dir,
            self.calibrated_dir, self.vis_dir, self.ply_dir
        ]
    
    def create_dirs(self):
        for d in self.all_dirs:
            os.makedirs(d, exist_ok=True)
    
    # ------------------------------------------------------------------
    # STEP 1: Extract RGB frames and masks
    # ------------------------------------------------------------------
    def step1_extract_frames(self):
        print(f"\n{'='*60}")
        print(f"STEP 1: Extracting RGB Frames and Masks")
        print(f"{'='*60}")
        
        os.makedirs(self.frames_raw_dir, exist_ok=True)
        os.makedirs(self.masks_raw_dir, exist_ok=True)
        
        for frame_idx in tqdm(range(self.num_frames), desc="Extracting frames"):
            frame_dir = os.path.join(self.seq_dir, str(frame_idx))
            
            # Copy RGB
            src_rgb = os.path.join(frame_dir, "rgbs", f"{self.view_idx:04d}.png")
            dst_rgb = os.path.join(self.frames_raw_dir, f"frame_{frame_idx:04d}.png")
            if os.path.exists(src_rgb):
                img = cv2.imread(src_rgb)
                cv2.imwrite(dst_rgb, img)
            
            # Copy Mask
            src_mask = os.path.join(frame_dir, "masks", f"{self.view_idx:04d}.png")
            dst_mask = os.path.join(self.masks_raw_dir, f"frame_{frame_idx:04d}.png")
            if os.path.exists(src_mask):
                mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(dst_mask, mask)
        
        print(f"  ✓ Saved raw frames to: {self.frames_raw_dir}")
        print(f"  ✓ Saved raw masks to: {self.masks_raw_dir}")
    
    # ------------------------------------------------------------------
    # STEP 2: Center crop and resize to 512x512
    # ------------------------------------------------------------------
    def step2_crop_and_resize(self):
        print(f"\n{'='*60}")
        print(f"STEP 2: Center Crop and Resize to 512x512")
        print(f"{'='*60}")
        
        os.makedirs(self.frames_512_dir, exist_ok=True)
        os.makedirs(self.masks_512_dir, exist_ok=True)
        
        for frame_idx in tqdm(range(self.num_frames), desc="Cropping and resizing"):
            # Process RGB
            src_rgb = os.path.join(self.frames_raw_dir, f"frame_{frame_idx:04d}.png")
            dst_rgb = os.path.join(self.frames_512_dir, f"frame_{frame_idx:04d}.png")
            if os.path.exists(src_rgb):
                img = cv2.imread(src_rgb)
                img_cropped = centre_crop(img, 1024)
                img_resized = cv2.resize(img_cropped, (512, 512))
                cv2.imwrite(dst_rgb, img_resized)
            
            # Process Mask
            src_mask = os.path.join(self.masks_raw_dir, f"frame_{frame_idx:04d}.png")
            dst_mask_png = os.path.join(self.masks_512_dir, f"frame_{frame_idx:04d}.png")
            dst_mask_npy = os.path.join(self.masks_512_dir, f"frame_{frame_idx:04d}.npy")
            if os.path.exists(src_mask):
                mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
                mask_cropped = centre_crop(mask, 1024)
                mask_resized = cv2.resize(mask_cropped, (512, 512), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(dst_mask_png, mask_resized)
                np.save(dst_mask_npy, mask_resized > 127)
        
        print(f"  ✓ Saved 512x512 frames to: {self.frames_512_dir}")
        print(f"  ✓ Saved 512x512 masks to: {self.masks_512_dir}")
    
    # ------------------------------------------------------------------
    # STEP 3: Run MoGe depth estimation
    # ------------------------------------------------------------------
    def step3_moge_depth(self):
        print(f"\n{'='*60}")
        print(f"STEP 3: Running MoGe Depth Estimation")
        print(f"{'='*60}")
        
        os.makedirs(self.moge_depth_dir, exist_ok=True)
        os.makedirs(self.moge_vis_dir, exist_ok=True)
        
        try:
            import torch
            from moge.model import MoGeModel
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  Using device: {device}")
            
            model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
            model.eval()
            
            moge_results = {}
            
            for frame_idx in tqdm(range(self.num_frames), desc="MoGe inference"):
                rgb_path = os.path.join(self.frames_512_dir, f"frame_{frame_idx:04d}.png")
                if not os.path.exists(rgb_path):
                    continue
                
                img = cv2.imread(rgb_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                img_tensor = torch.from_numpy(img).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).to(device)
                
                with torch.no_grad():
                    output = model.infer(img_tensor)
                
                depth = output['depth'].cpu().numpy()
                mask = output['mask'].cpu().numpy()
                intrinsics = output['intrinsics'].cpu().numpy()
                
                np.save(os.path.join(self.moge_depth_dir, f"frame_{frame_idx:04d}.npy"), depth)
                depth_vis = depth_to_color(depth)
                cv2.imwrite(os.path.join(self.moge_vis_dir, f"frame_{frame_idx:04d}.png"), depth_vis)
                
                moge_results[f"frame_{frame_idx:04d}"] = {
                    'depth': depth, 'mask': mask, 'intrinsics': intrinsics
                }
            
            np.save(os.path.join(self.repro_dir, "moge_results.npy"), moge_results)
            print(f"  ✓ Saved MoGe depth to: {self.moge_depth_dir}")
            
        except ImportError as e:
            print(f"  ✗ Error: Could not import MoGe: {e}")
            return False
        return True
    
    # ------------------------------------------------------------------
    # STEP 3b: Load existing MoGe depth
    # ------------------------------------------------------------------
    def step3b_load_existing_moge(self, existing_path=None):
        print(f"\n{'='*60}")
        print(f"STEP 3b: Loading Existing MoGe Depth")
        print(f"{'='*60}")
        
        if existing_path is None:
            # Try to find existing file
            # Map sequence names: 0012_09 -> DNA_02, etc.
            existing_path = EXISTING_MOGE_PATTERN.format(seq="DNA_02")
        
        if not os.path.exists(existing_path):
            print(f"  ✗ Error: Existing MoGe file not found at {existing_path}")
            return False
        
        os.makedirs(self.moge_depth_dir, exist_ok=True)
        os.makedirs(self.moge_vis_dir, exist_ok=True)
        
        print(f"  Loading from: {existing_path}")
        existing_data = np.load(existing_path, allow_pickle=True).item()
        existing_keys = sorted(list(existing_data.keys()))
        print(f"  Found {len(existing_keys)} frames in existing file")
        
        moge_results = {}
        
        for frame_idx in tqdm(range(self.num_frames), desc="Processing MoGe data"):
            matching_key = None
            for key in existing_keys:
                if f"{frame_idx:04d}" in key or f"/{frame_idx}/" in key:
                    matching_key = key
                    break
            
            if matching_key is None:
                continue
            
            data = existing_data[matching_key]
            depth = data['depth']
            mask = data['mask']
            intrinsics = data.get('intrinsics', np.eye(3))
            
            np.save(os.path.join(self.moge_depth_dir, f"frame_{frame_idx:04d}.npy"), depth)
            depth_vis = depth_to_color(depth)
            cv2.imwrite(os.path.join(self.moge_vis_dir, f"frame_{frame_idx:04d}.png"), depth_vis)
            
            moge_results[f"frame_{frame_idx:04d}"] = {
                'depth': depth, 'mask': mask, 'intrinsics': intrinsics
            }
        
        np.save(os.path.join(self.repro_dir, "moge_results.npy"), moge_results)
        print(f"  ✓ Saved {len(moge_results)} MoGe depth frames to: {self.moge_depth_dir}")
        return True
    
    # ------------------------------------------------------------------
    # STEP 4: Render PCL depth at raw resolution
    # ------------------------------------------------------------------
    def step4_render_pcl_depth(self):
        print(f"\n{'='*60}")
        print(f"STEP 4: Rendering PCL Depth at Raw Resolution (1024x1224)")
        print(f"{'='*60}")
        
        os.makedirs(self.rendered_raw_dir, exist_ok=True)
        
        for frame_idx in tqdm(range(self.num_frames), desc="Rendering PCL depth"):
            frame_dir = os.path.join(self.seq_dir, str(frame_idx))
            
            ply_path = os.path.join(frame_dir, "point_cloud", "iteration_20000", "point_cloud.ply")
            if not os.path.exists(ply_path):
                ply_path = os.path.join(frame_dir, "point_cloud", "iteration_15000", "point_cloud.ply")
            if not os.path.exists(ply_path):
                continue
            
            cameras_path = os.path.join(frame_dir, "cameras.json")
            cam = get_camera_for_view(cameras_path, self.view_idx)
            if cam is None:
                continue
            
            pcd = o3d.io.read_point_cloud(ply_path)
            points_world = np.asarray(pcd.points)
            
            width, height = cam['width'], cam['height']
            fx, fy = cam['fx'], cam['fy']
            cx, cy = width / 2.0, height / 2.0
            
            pos = np.array(cam['position'])
            rot = np.array(cam['rotation'])
            R_cw = rot.T
            t_cw = -R_cw @ pos
            
            points_cam = (points_world @ R_cw.T) + t_cw
            valid_z = points_cam[:, 2] > 0
            points_cam = points_cam[valid_z]
            
            if len(points_cam) == 0:
                continue
            
            z = points_cam[:, 2]
            u = (points_cam[:, 0] * fx / z) + cx
            v = (points_cam[:, 1] * fy / z) + cy
            
            valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            u, v, z = u[valid_uv], v[valid_uv], z[valid_uv]
            
            if len(z) == 0:
                continue
            
            sort_idx = np.argsort(z)[::-1]
            u_sorted, v_sorted, z_sorted = u[sort_idx], v[sort_idx], z[sort_idx]
            
            depth_map = np.zeros((height, width), dtype=np.float32)
            u_int = np.round(u_sorted).astype(int)
            v_int = np.round(v_sorted).astype(int)
            
            for i in range(len(z_sorted)):
                cv2.circle(depth_map, (u_int[i], v_int[i]), 2, float(z_sorted[i]), -1)
            
            # Fill holes
            mask_valid = (depth_map > 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_closed = cv2.morphologyEx(mask_valid, cv2.MORPH_CLOSE, kernel)
            holes = (mask_closed - mask_valid).astype(np.uint8)
            if np.sum(holes) > 0:
                depth_mm = (depth_map * 1000).astype(np.uint16)
                depth_inpainted = cv2.inpaint(depth_mm, holes, 3, cv2.INPAINT_TELEA)
                depth_map = depth_inpainted.astype(np.float32) / 1000.0
            depth_map = cv2.medianBlur(depth_map, 5)
            depth_map = np.where(mask_closed > 0, depth_map, 0.0)
            
            np.save(os.path.join(self.rendered_raw_dir, f"frame_{frame_idx:04d}.npy"), depth_map)
            depth_vis = depth_to_color(depth_map)
            cv2.imwrite(os.path.join(self.rendered_raw_dir, f"frame_{frame_idx:04d}_vis.png"), depth_vis)
        
        print(f"  ✓ Saved raw rendered depth to: {self.rendered_raw_dir}")
    
    # ------------------------------------------------------------------
    # STEP 5: Center crop and resize rendered depth
    # ------------------------------------------------------------------
    def step5_crop_rendered_depth(self):
        print(f"\n{'='*60}")
        print(f"STEP 5: Center Crop and Resize Rendered Depth to 512x512")
        print(f"{'='*60}")
        
        os.makedirs(self.rendered_512_dir, exist_ok=True)
        
        for frame_idx in tqdm(range(self.num_frames), desc="Processing rendered depth"):
            src_path = os.path.join(self.rendered_raw_dir, f"frame_{frame_idx:04d}.npy")
            if not os.path.exists(src_path):
                continue
            
            depth = np.load(src_path)
            depth_cropped = centre_crop(depth, 1024)
            depth_resized = cv2.resize(depth_cropped, (512, 512), interpolation=cv2.INTER_NEAREST)
            
            np.save(os.path.join(self.rendered_512_dir, f"frame_{frame_idx:04d}.npy"), depth_resized)
            depth_vis = depth_to_color(depth_resized)
            cv2.imwrite(os.path.join(self.rendered_512_dir, f"frame_{frame_idx:04d}_vis.png"), depth_vis)
        
        print(f"  ✓ Saved 512x512 rendered depth to: {self.rendered_512_dir}")
    
    # ------------------------------------------------------------------
    # STEP 6: Calibrate MoGe depth
    # ------------------------------------------------------------------
    def step6_calibrate(self):
        print(f"\n{'='*60}")
        print(f"STEP 6: Calibrating MoGe Depth")
        print(f"{'='*60}")
        
        os.makedirs(self.calibrated_dir, exist_ok=True)
        
        moge_path = os.path.join(self.repro_dir, "moge_results.npy")
        if not os.path.exists(moge_path):
            print(f"  ✗ Error: MoGe results not found. Run step 3 first.")
            return None
        
        moge_data = np.load(moge_path, allow_pickle=True).item()
        scale_factors = []
        calibrated_depths = {}
        
        for frame_idx in tqdm(range(self.num_frames), desc="Calibrating"):
            gt_path = os.path.join(self.rendered_512_dir, f"frame_{frame_idx:04d}.npy")
            mask_path = os.path.join(self.masks_512_dir, f"frame_{frame_idx:04d}.npy")
            moge_key = f"frame_{frame_idx:04d}"
            
            if not os.path.exists(gt_path) or moge_key not in moge_data:
                continue
            
            gt_depth = np.load(gt_path)
            moge_depth = moge_data[moge_key]['depth']
            moge_mask = moge_data[moge_key]['mask'].astype(bool)
            
            if os.path.exists(mask_path):
                fg_mask = np.load(mask_path)
            else:
                fg_mask = gt_depth > 0
            
            overlap = fg_mask & moge_mask & (gt_depth > 0) & (moge_depth > 0)
            if overlap.sum() < 100:
                continue
            
            gt_median = np.median(gt_depth[overlap])
            moge_median = np.median(moge_depth[overlap])
            scale = gt_median / moge_median if moge_median > 0 else 1.0
            scale_factors.append(scale)
            
            calibrated = moge_depth.copy() * scale
            calibrated_depths[frame_idx] = {
                'depth': calibrated, 'mask': moge_mask, 'fg_mask': fg_mask,
                'scale': scale, 'intrinsics': moge_data[moge_key]['intrinsics']
            }
            
            np.save(os.path.join(self.calibrated_dir, f"frame_{frame_idx:04d}.npy"), calibrated)
            depth_vis = depth_to_color(calibrated)
            cv2.imwrite(os.path.join(self.calibrated_dir, f"frame_{frame_idx:04d}_vis.png"), depth_vis)
        
        if scale_factors:
            print(f"\n  Scale factor statistics:")
            print(f"    Mean: {np.mean(scale_factors):.4f}")
            print(f"    Std:  {np.std(scale_factors):.4f}")
        
        print(f"  ✓ Saved calibrated depth to: {self.calibrated_dir}")
        return calibrated_depths
    
    # ------------------------------------------------------------------
    # STEP 7: Create visualizations
    # ------------------------------------------------------------------
    def step7_visualize(self, calibrated_depths):
        print(f"\n{'='*60}")
        print(f"STEP 7: Creating Visualizations")
        print(f"{'='*60}")
        
        os.makedirs(self.vis_dir, exist_ok=True)
        moge_data = np.load(os.path.join(self.repro_dir, "moge_results.npy"), allow_pickle=True).item()
        
        for frame_idx in tqdm(list(calibrated_depths.keys()), desc="Creating visualizations"):
            rgb_path = os.path.join(self.frames_512_dir, f"frame_{frame_idx:04d}.png")
            if not os.path.exists(rgb_path):
                continue
            
            rgb = cv2.imread(rgb_path)
            gt_depth = np.load(os.path.join(self.rendered_512_dir, f"frame_{frame_idx:04d}.npy"))
            moge_depth = moge_data[f"frame_{frame_idx:04d}"]['depth']
            calibrated = calibrated_depths[frame_idx]['depth']
            scale = calibrated_depths[frame_idx]['scale']
            
            gt_valid = gt_depth[gt_depth > 0]
            vmin, vmax = (gt_valid.min(), gt_valid.max()) if gt_valid.size > 0 else (0, 1)
            
            gt_vis = depth_to_color(gt_depth, vmin, vmax)
            moge_vis = depth_to_color(moge_depth)
            calibrated_vis = depth_to_color(calibrated, vmin, vmax)
            
            cv2.putText(rgb, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(gt_vis, "GT Rendered", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(moge_vis, "MoGe (raw)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(calibrated_vis, f"Calibrated (s={scale:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            grid = np.hstack([rgb, gt_vis, moge_vis, calibrated_vis])
            cv2.imwrite(os.path.join(self.vis_dir, f"frame_{frame_idx:04d}.png"), grid)
        
        print(f"  ✓ Saved visualizations to: {self.vis_dir}")
    
    # ------------------------------------------------------------------
    # STEP 8: Create unprojected PLY comparisons
    # ------------------------------------------------------------------
    def step8_unproject_ply(self, calibrated_depths, frame_indices=None):
        print(f"\n{'='*60}")
        print(f"STEP 8: Creating Unprojected PLY Comparisons")
        print(f"{'='*60}")
        
        os.makedirs(self.ply_dir, exist_ok=True)
        
        if frame_indices is None:
            frame_indices = [0, 30, 60, 90, 120]
        
        moge_data = np.load(os.path.join(self.repro_dir, "moge_results.npy"), allow_pickle=True).item()
        
        for frame_idx in tqdm(frame_indices, desc="Unprojecting"):
            if frame_idx not in calibrated_depths:
                continue
            
            data = calibrated_depths[frame_idx]
            moge_intr = data['intrinsics']
            H, W = 512, 512
            fx, fy = moge_intr[0, 0] * W, moge_intr[1, 1] * H
            cx, cy = moge_intr[0, 2] * W, moge_intr[1, 2] * H
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
            gt_depth = np.load(os.path.join(self.rendered_512_dir, f"frame_{frame_idx:04d}.npy"))
            moge_depth = moge_data[f"frame_{frame_idx:04d}"]['depth']
            calibrated = data['depth']
            fg_mask = data['fg_mask']
            moge_mask = data['mask']
            bg_mask = moge_mask & ~fg_mask
            
            gt_pts, _ = unproject_depth_to_3d(gt_depth, intrinsics, fg_mask)
            calib_fg_pts, _ = unproject_depth_to_3d(calibrated, intrinsics, fg_mask)
            calib_bg_pts, _ = unproject_depth_to_3d(calibrated, intrinsics, bg_mask)
            
            gt_pcd = create_colored_pointcloud(gt_pts, [0, 1, 0])
            calib_fg_pcd = create_colored_pointcloud(calib_fg_pts, [0, 0.5, 1])
            calib_bg_pcd = create_colored_pointcloud(calib_bg_pts, [0, 0, 1])
            
            o3d.io.write_point_cloud(os.path.join(self.ply_dir, f"frame_{frame_idx:04d}_gt.ply"), gt_pcd)
            o3d.io.write_point_cloud(os.path.join(self.ply_dir, f"frame_{frame_idx:04d}_calibrated_fg.ply"), calib_fg_pcd)
            o3d.io.write_point_cloud(os.path.join(self.ply_dir, f"frame_{frame_idx:04d}_calibrated_bg.ply"), calib_bg_pcd)
            o3d.io.write_point_cloud(os.path.join(self.ply_dir, f"frame_{frame_idx:04d}_full_scene.ply"), gt_pcd + calib_bg_pcd)
        
        print(f"  ✓ Saved PLY files to: {self.ply_dir}")
    
    # ------------------------------------------------------------------
    # STEP 9: Create combined calibrated npy file
    # ------------------------------------------------------------------
    def step9_create_combined_npy(self):
        print(f"\n{'='*60}")
        print(f"STEP 9: Creating Combined Calibrated NPY File")
        print(f"{'='*60}")
        
        moge_path = os.path.join(self.repro_dir, "moge_results.npy")
        output_path = os.path.join(self.repro_dir, "moge_calibrated.npy")
        
        if not os.path.exists(moge_path):
            print(f"  ✗ Error: MoGe results not found")
            return False
        
        moge_data = np.load(moge_path, allow_pickle=True).item()
        calibrated_output = {}
        
        for frame_idx in tqdm(range(self.num_frames), desc="Creating combined file"):
            moge_key = f"frame_{frame_idx:04d}"
            calib_path = os.path.join(self.calibrated_dir, f"frame_{frame_idx:04d}.npy")
            
            if not os.path.exists(calib_path) or moge_key not in moge_data:
                continue
            
            calibrated_depth = np.load(calib_path)
            calibrated_output[moge_key] = {
                'depth': calibrated_depth,
                'mask': moge_data[moge_key]['mask'],
                'intrinsics': moge_data[moge_key]['intrinsics']
            }
        
        np.save(output_path, calibrated_output)
        print(f"  ✓ Saved combined file to: {output_path}")
        print(f"  ✓ Contains {len(calibrated_output)} frames")
        return True
    
    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    def run(self, steps, load_moge=False, frame_indices=None):
        # If fast mode, use the streamlined method
        if self.fast_mode:
            return self.run_fast(load_moge=load_moge)
        
        print(f"\n{'='*60}")
        print(f"Processing {self.seq_name} view {self.view_idx}")
        print(f"Output: {self.repro_dir}")
        print(f"{'='*60}")
        
        self.create_dirs()
        calibrated_depths = None
        
        if 1 in steps:
            self.step1_extract_frames()
        if 2 in steps:
            self.step2_crop_and_resize()
        if 3 in steps:
            if load_moge:
                self.step3b_load_existing_moge()
            else:
                self.step3_moge_depth()
        if 4 in steps:
            self.step4_render_pcl_depth()
        if 5 in steps:
            self.step5_crop_rendered_depth()
        if 6 in steps:
            calibrated_depths = self.step6_calibrate()
        if 7 in steps:
            if calibrated_depths is None:
                calibrated_depths = self.step6_calibrate()
            if calibrated_depths:
                self.step7_visualize(calibrated_depths)
        if 8 in steps:
            if calibrated_depths is None:
                calibrated_depths = self.step6_calibrate()
            if calibrated_depths:
                self.step8_unproject_ply(calibrated_depths, frame_indices)
        if 9 in steps:
            self.step9_create_combined_npy()
        
        print(f"\n{'='*60}")
        print(f"DONE! Output: {self.repro_dir}")
        print(f"{'='*60}")
    
    # ------------------------------------------------------------------
    # Run FAST: Skip all intermediate files, only produce final output
    # ------------------------------------------------------------------
    def run_fast(self, load_moge=False):
        """Fast mode: process in memory, only save moge_calibrated.npy"""
        print(f"\n{'='*60}")
        print(f"FAST MODE: {self.seq_name} view {self.view_idx}")
        print(f"Output: {self.repro_dir}/moge_calibrated.npy")
        print(f"{'='*60}")
        
        os.makedirs(self.repro_dir, exist_ok=True)
        
        # Get MoGe depth - either load existing or run inference
        moge_data_dict = {}
        
        if load_moge:
            # Try to load existing MoGe data (only works for original view)
            existing_path = EXISTING_MOGE_PATTERN.format(seq="DNA_02")
            if not os.path.exists(existing_path):
                print(f"  ✗ Error: MoGe file not found at {existing_path}")
                print(f"  ⚠ Note: Existing MoGe data is only for the original view!")
                print(f"  ⚠ For other views, remove --load_moge to run MoGe inference.")
                return False
            
            print(f"  Loading existing MoGe from: {existing_path}")
            print(f"  ⚠ Warning: This data was created for a specific view!")
            existing_data = np.load(existing_path, allow_pickle=True).item()
            existing_keys = sorted(list(existing_data.keys()))
            
            for frame_idx in range(self.num_frames):
                matching_key = None
                for key in existing_keys:
                    if f"{frame_idx:04d}" in key:
                        matching_key = key
                        break
                if matching_key:
                    moge_data_dict[frame_idx] = existing_data[matching_key]
        else:
            # Run MoGe inference for this specific view
            print(f"  Running MoGe inference for view {self.view_idx}...")
            try:
                import torch
                from moge.model import MoGeModel
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"  Using device: {device}")
                
                model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
                model.eval()
                
                for frame_idx in tqdm(range(self.num_frames), desc="MoGe inference"):
                    frame_dir = os.path.join(self.seq_dir, str(frame_idx))
                    rgb_path = os.path.join(frame_dir, "rgbs", f"{self.view_idx:04d}.png")
                    if not os.path.exists(rgb_path):
                        continue
                    
                    # Load and preprocess image
                    img = cv2.imread(rgb_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_cropped = centre_crop(img, 1024)
                    img_resized = cv2.resize(img_cropped, (512, 512))
                    
                    img_tensor = torch.from_numpy(img_resized).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1).to(device)
                    
                    with torch.no_grad():
                        output = model.infer(img_tensor)
                    
                    moge_data_dict[frame_idx] = {
                        'depth': output['depth'].cpu().numpy(),
                        'mask': output['mask'].cpu().numpy(),
                        'intrinsics': output['intrinsics'].cpu().numpy()
                    }
                    
            except ImportError as e:
                print(f"  ✗ Error: Could not import MoGe: {e}")
                return False
        
        print(f"  Got MoGe data for {len(moge_data_dict)} frames")
        
        calibrated_output = {}
        scale_factors = []
        
        for frame_idx in tqdm(range(self.num_frames), desc="Calibrating"):
            if frame_idx not in moge_data_dict:
                continue
            
            moge_data = moge_data_dict[frame_idx]
            moge_depth = moge_data['depth']
            moge_mask = moge_data['mask'].astype(bool)
            intrinsics = moge_data.get('intrinsics', np.eye(3))
            
            # Load source data directly
            frame_dir = os.path.join(self.seq_dir, str(frame_idx))
            
            # Get mask (for foreground)
            mask_path = os.path.join(frame_dir, "masks", f"{self.view_idx:04d}.png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_cropped = centre_crop(mask, 1024)
                mask_resized = cv2.resize(mask_cropped, (512, 512), interpolation=cv2.INTER_NEAREST)
                fg_mask = mask_resized > 127
            else:
                fg_mask = None
            
            # Render GT depth directly (in memory)
            ply_path = os.path.join(frame_dir, "point_cloud", "iteration_20000", "point_cloud.ply")
            if not os.path.exists(ply_path):
                ply_path = os.path.join(frame_dir, "point_cloud", "iteration_15000", "point_cloud.ply")
            if not os.path.exists(ply_path):
                continue
            
            cameras_path = os.path.join(frame_dir, "cameras.json")
            cam = get_camera_for_view(cameras_path, self.view_idx)
            if cam is None:
                continue
            
            # Render depth
            pcd = o3d.io.read_point_cloud(ply_path)
            points_world = np.asarray(pcd.points)
            
            width, height = cam['width'], cam['height']
            fx, fy = cam['fx'], cam['fy']
            cx, cy = width / 2.0, height / 2.0
            
            pos = np.array(cam['position'])
            rot = np.array(cam['rotation'])
            R_cw = rot.T
            t_cw = -R_cw @ pos
            
            points_cam = (points_world @ R_cw.T) + t_cw
            valid_z = points_cam[:, 2] > 0
            points_cam = points_cam[valid_z]
            
            if len(points_cam) == 0:
                continue
            
            z = points_cam[:, 2]
            u = (points_cam[:, 0] * fx / z) + cx
            v = (points_cam[:, 1] * fy / z) + cy
            
            valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            u, v, z = u[valid_uv], v[valid_uv], z[valid_uv]
            
            if len(z) == 0:
                continue
            
            sort_idx = np.argsort(z)[::-1]
            u_sorted, v_sorted, z_sorted = u[sort_idx], v[sort_idx], z[sort_idx]
            
            depth_map = np.zeros((height, width), dtype=np.float32)
            u_int = np.round(u_sorted).astype(int)
            v_int = np.round(v_sorted).astype(int)
            
            for i in range(len(z_sorted)):
                cv2.circle(depth_map, (u_int[i], v_int[i]), 2, float(z_sorted[i]), -1)
            
            # Center crop and resize
            gt_depth = centre_crop(depth_map, 1024)
            gt_depth = cv2.resize(gt_depth, (512, 512), interpolation=cv2.INTER_NEAREST)
            
            if fg_mask is None:
                fg_mask = gt_depth > 0
            
            # Calibrate
            overlap = fg_mask & moge_mask & (gt_depth > 0) & (moge_depth > 0)
            if overlap.sum() < 100:
                continue
            
            gt_median = np.median(gt_depth[overlap])
            moge_median = np.median(moge_depth[overlap])
            scale = gt_median / moge_median if moge_median > 0 else 1.0
            scale_factors.append(scale)
            
            calibrated = moge_depth * scale
            
            calibrated_output[f"frame_{frame_idx:04d}"] = {
                'depth': calibrated,
                'mask': moge_mask,
                'intrinsics': intrinsics
            }
        
        # Save final output
        output_path = os.path.join(self.repro_dir, "moge_calibrated.npy")
        np.save(output_path, calibrated_output)
        
        print(f"\n  Scale factor: mean={np.mean(scale_factors):.4f}, std={np.std(scale_factors):.4f}")
        print(f"  ✓ Saved: {output_path}")
        print(f"  ✓ Contains {len(calibrated_output)} frames")
        return True


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Process DNA-Rendering sequence depth calibration")
    parser.add_argument('--seq', type=str, required=True, help='Sequence name (e.g., 0012_09)')
    parser.add_argument('--view', type=int, default=None, help='Single view index (e.g., 22)')
    parser.add_argument('--views', type=str, default=None, help='Multiple views: comma-separated (e.g., 0,10,22) or "all"')
    parser.add_argument('--step', type=str, default='all', help='Steps to run: all, or comma-separated (e.g., 1,2,3)')
    parser.add_argument('--frames', type=str, default='0,30,60,90,120', help='Frame indices for PLY export')
    parser.add_argument('--load_moge', action='store_true', help='Load existing MoGe data instead of running inference')
    parser.add_argument('--fast', action='store_true', help='Fast mode: skip intermediate files, only produce moge_calibrated.npy')
    args = parser.parse_args()
    
    # Parse steps
    if args.step.lower() == 'all':
        steps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        steps = [int(s.strip()) for s in args.step.split(',')]
    
    # Parse frame indices
    frame_indices = [int(x) for x in args.frames.split(',')]
    
    # Parse views
    if args.views is not None:
        if args.views.lower() == 'all':
            views = list(range(NUM_VIEWS))
        else:
            views = [int(v.strip()) for v in args.views.split(',')]
    elif args.view is not None:
        views = [args.view]
    else:
        print("Error: Must specify --view or --views")
        return
    
    print(f"\n{'#'*60}")
    print(f"# Processing {len(views)} view(s) for sequence {args.seq}")
    print(f"{'#'*60}")
    
    for view_idx in views:
        pipeline = DepthCalibrationPipeline(args.seq, view_idx, fast_mode=args.fast)
        pipeline.run(steps, load_moge=args.load_moge, frame_indices=frame_indices)


def process_single_view(args_tuple):
    """Worker function for multiprocessing with GPU assignment."""
    seq_name, view_idx, steps, load_moge, frame_indices, fast_mode, gpu_id, output_base = args_tuple
    
    # Set GPU for this worker BEFORE importing torch
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        pipeline = DepthCalibrationPipeline(seq_name, view_idx, fast_mode=fast_mode, output_base=output_base)
        pipeline.run(steps, load_moge=load_moge, frame_indices=frame_indices)
        return (view_idx, True, None, gpu_id)
    except Exception as e:
        import traceback
        return (view_idx, False, traceback.format_exc(), gpu_id)


def main_parallel():
    """Main function with multiprocessing support."""
    import multiprocessing as mp
    
    parser = argparse.ArgumentParser(description="Process DNA-Rendering sequence depth calibration (parallel)")
    parser.add_argument('--seq', type=str, required=True, help='Sequence name (e.g., 0012_09)')
    parser.add_argument('--view', type=int, default=None, help='Single view index (e.g., 22)')
    parser.add_argument('--views', type=str, default=None, help='Multiple views: comma-separated (e.g., 0,10,22) or "all"')
    parser.add_argument('--step', type=str, default='all', help='Steps to run: all, or comma-separated (e.g., 1,2,3)')
    parser.add_argument('--frames', type=str, default='0,30,60,90,120', help='Frame indices for PLY export')
    parser.add_argument('--load_moge', action='store_true', help='Load existing MoGe data instead of running inference')
    parser.add_argument('--fast', action='store_true', help='Fast mode: skip intermediate files, only produce moge_calibrated.npy')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (default: 1, max: num GPUs)')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6', help='Comma-separated GPU IDs to use (default: 0,1,2,3,4,5,6)')
    parser.add_argument('--output_dir', type=str, default=None, help='Custom output directory (default: reproduction/)')
    args = parser.parse_args()
    
    # Parse output directory
    output_base = args.output_dir or OUTPUT_BASE
    
    # Parse steps
    if args.step.lower() == 'all':
        steps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        steps = [int(s.strip()) for s in args.step.split(',')]
    
    # Parse frame indices
    frame_indices = [int(x) for x in args.frames.split(',')]
    
    # Parse views
    if args.views is not None:
        if args.views.lower() == 'all':
            views = list(range(NUM_VIEWS))
        else:
            views = [int(v.strip()) for v in args.views.split(',')]
    elif args.view is not None:
        views = [args.view]
    else:
        print("Error: Must specify --view or --views")
        return
    
    print(f"\n{'#'*60}")
    print(f"# Processing {len(views)} view(s) for sequence {args.seq}")
    print(f"# Workers: {args.workers}")
    print(f"# Fast mode: {args.fast}")
    print(f"# GPUs: {args.gpus}")
    print(f"{'#'*60}")
    
    if args.workers <= 1:
        # Sequential processing
        for view_idx in views:
            pipeline = DepthCalibrationPipeline(args.seq, view_idx, fast_mode=args.fast, output_base=output_base)
            pipeline.run(steps, load_moge=args.load_moge, frame_indices=frame_indices)
    else:
        # Parallel processing with GPU assignment
        num_gpus = len(args.gpus.split(',')) if args.gpus else 7
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')] if args.gpus else list(range(num_gpus))
        
        # Limit workers to number of GPUs
        actual_workers = min(args.workers, len(gpu_ids))
        if actual_workers < args.workers:
            print(f"⚠ Limiting workers to {actual_workers} (number of GPUs)")
        
        # Assign GPU to each view (round-robin)
        work_items = []
        for i, view_idx in enumerate(views):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            work_items.append(
                (args.seq, view_idx, steps, args.load_moge, frame_indices, args.fast, gpu_id, output_base)
            )
        
        print(f"\nStarting {actual_workers} parallel workers on GPUs {gpu_ids}...")
        print(f"Processing {len(views)} views...")
        
        # Use spawn to avoid CUDA issues
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=actual_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_view, work_items),
                total=len(work_items),
                desc="Processing views"
            ))
        
        # Report results
        success = sum(1 for r in results if r[1])
        failed = [(r[0], r[2]) for r in results if not r[1]]
        
        print(f"\n{'='*60}")
        print(f"PARALLEL PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"  ✓ Successful: {success}/{len(views)}")
        if failed:
            print(f"  ✗ Failed: {len(failed)}")
            for v, e in failed[:5]:  # Show first 5 errors
                print(f"    View {v}: {e[:100]}...")
    
    print(f"\n{'='*60}")
    print(f"ALL DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Use parallel main by default
    main_parallel()


