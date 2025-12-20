#!/usr/bin/env python3
"""
Process 0012_09 sequence view 22 - Complete Depth Calibration Pipeline

Step-by-step workflow:
1. Extract RGB frames to reproduction/frames_raw/
2. Center crop and resize to 512x512 -> reproduction/frames_512/
3. Run MoGe depth estimation -> reproduction/moge_depth/
4. Render PCL depth at raw resolution (1024x1224) -> reproduction/rendered_depth_raw/
5. Center crop and resize rendered depth -> reproduction/rendered_depth_512/
6. Calibrate MoGe using rendered depth -> reproduction/calibrated_depth/
7. Create visualizations -> reproduction/visualizations/
8. Create unprojected PLY comparisons -> reproduction/unprojected_ply/

Usage:
    python process_0012_09_view22.py --step all     # Run all steps
    python process_0012_09_view22.py --step 1       # Run only step 1
    python process_0012_09_view22.py --step 1,2,3   # Run steps 1, 2, 3
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
SEQ_NAME = "0012_09"
VIEW_IDX = 22
NUM_FRAMES = 150

# Source data
BASE_DATA_DIR = "/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/Part1"
SEQ_DIR = os.path.join(BASE_DATA_DIR, SEQ_NAME)

# Output directory with clean structure
REPRO_DIR = f"/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/reproduction/{SEQ_NAME}_view{VIEW_IDX}"

# Subfolder structure
FRAMES_RAW_DIR = os.path.join(REPRO_DIR, "1_frames_raw")           # Step 1
FRAMES_512_DIR = os.path.join(REPRO_DIR, "2_frames_512")           # Step 2
MASKS_RAW_DIR = os.path.join(REPRO_DIR, "1_masks_raw")             # Step 1
MASKS_512_DIR = os.path.join(REPRO_DIR, "2_masks_512")             # Step 2
MOGE_DEPTH_DIR = os.path.join(REPRO_DIR, "3_moge_depth")           # Step 3
MOGE_VIS_DIR = os.path.join(REPRO_DIR, "3_moge_vis")               # Step 3
RENDERED_RAW_DIR = os.path.join(REPRO_DIR, "4_rendered_depth_raw") # Step 4
RENDERED_512_DIR = os.path.join(REPRO_DIR, "5_rendered_depth_512") # Step 5
CALIBRATED_DIR = os.path.join(REPRO_DIR, "6_calibrated_depth")     # Step 6
VIS_DIR = os.path.join(REPRO_DIR, "7_visualizations")              # Step 7
PLY_DIR = os.path.join(REPRO_DIR, "8_unprojected_ply")             # Step 8

ALL_DIRS = [REPRO_DIR, FRAMES_RAW_DIR, FRAMES_512_DIR, MASKS_RAW_DIR, MASKS_512_DIR,
            MOGE_DEPTH_DIR, MOGE_VIS_DIR, RENDERED_RAW_DIR, RENDERED_512_DIR,
            CALIBRATED_DIR, VIS_DIR, PLY_DIR]

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
# STEP 1: Extract RGB frames and masks
# ======================================================================
def step1_extract_frames():
    """Extract RGB frames and masks from source to reproduction folder."""
    print("\n" + "="*60)
    print("STEP 1: Extracting RGB Frames and Masks")
    print("="*60)
    
    os.makedirs(FRAMES_RAW_DIR, exist_ok=True)
    os.makedirs(MASKS_RAW_DIR, exist_ok=True)
    
    for frame_idx in tqdm(range(NUM_FRAMES), desc="Extracting frames"):
        frame_dir = os.path.join(SEQ_DIR, str(frame_idx))
        
        # Copy RGB
        src_rgb = os.path.join(frame_dir, "rgbs", f"{VIEW_IDX:04d}.png")
        dst_rgb = os.path.join(FRAMES_RAW_DIR, f"frame_{frame_idx:04d}.png")
        if os.path.exists(src_rgb):
            img = cv2.imread(src_rgb)
            cv2.imwrite(dst_rgb, img)
        
        # Copy Mask
        src_mask = os.path.join(frame_dir, "masks", f"{VIEW_IDX:04d}.png")
        dst_mask = os.path.join(MASKS_RAW_DIR, f"frame_{frame_idx:04d}.png")
        if os.path.exists(src_mask):
            mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(dst_mask, mask)
    
    print(f"  ✓ Saved raw frames to: {FRAMES_RAW_DIR}")
    print(f"  ✓ Saved raw masks to: {MASKS_RAW_DIR}")


# ======================================================================
# STEP 2: Center crop and resize to 512x512
# ======================================================================
def step2_crop_and_resize():
    """Center crop (1024x1024) and resize to 512x512."""
    print("\n" + "="*60)
    print("STEP 2: Center Crop and Resize to 512x512")
    print("="*60)
    
    os.makedirs(FRAMES_512_DIR, exist_ok=True)
    os.makedirs(MASKS_512_DIR, exist_ok=True)
    
    for frame_idx in tqdm(range(NUM_FRAMES), desc="Cropping and resizing"):
        # Process RGB
        src_rgb = os.path.join(FRAMES_RAW_DIR, f"frame_{frame_idx:04d}.png")
        dst_rgb = os.path.join(FRAMES_512_DIR, f"frame_{frame_idx:04d}.png")
        if os.path.exists(src_rgb):
            img = cv2.imread(src_rgb)
            img_cropped = centre_crop(img, 1024)
            img_resized = cv2.resize(img_cropped, (512, 512))
            cv2.imwrite(dst_rgb, img_resized)
        
        # Process Mask
        src_mask = os.path.join(MASKS_RAW_DIR, f"frame_{frame_idx:04d}.png")
        dst_mask_png = os.path.join(MASKS_512_DIR, f"frame_{frame_idx:04d}.png")
        dst_mask_npy = os.path.join(MASKS_512_DIR, f"frame_{frame_idx:04d}.npy")
        if os.path.exists(src_mask):
            mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
            mask_cropped = centre_crop(mask, 1024)
            mask_resized = cv2.resize(mask_cropped, (512, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(dst_mask_png, mask_resized)
            np.save(dst_mask_npy, mask_resized > 127)  # Boolean mask
    
    print(f"  ✓ Saved 512x512 frames to: {FRAMES_512_DIR}")
    print(f"  ✓ Saved 512x512 masks to: {MASKS_512_DIR}")


# ======================================================================
# STEP 3: Run MoGe depth estimation
# ======================================================================
def step3_moge_depth():
    """Run MoGe depth estimation on 512x512 frames."""
    print("\n" + "="*60)
    print("STEP 3: Running MoGe Depth Estimation")
    print("="*60)
    
    os.makedirs(MOGE_DEPTH_DIR, exist_ok=True)
    os.makedirs(MOGE_VIS_DIR, exist_ok=True)
    
    try:
        import torch
        from moge.model import MoGeModel
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")
        
        model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
        model.eval()
        
        moge_results = {}
        
        for frame_idx in tqdm(range(NUM_FRAMES), desc="MoGe inference"):
            rgb_path = os.path.join(FRAMES_512_DIR, f"frame_{frame_idx:04d}.png")
            if not os.path.exists(rgb_path):
                continue
            
            # Load image
            img = cv2.imread(rgb_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).to(device)
            
            with torch.no_grad():
                output = model.infer(img_tensor)
            
            depth = output['depth'].cpu().numpy()
            mask = output['mask'].cpu().numpy()
            intrinsics = output['intrinsics'].cpu().numpy()
            
            # Save depth as npy
            np.save(os.path.join(MOGE_DEPTH_DIR, f"frame_{frame_idx:04d}.npy"), depth)
            
            # Save visualization
            depth_vis = depth_to_color(depth)
            cv2.imwrite(os.path.join(MOGE_VIS_DIR, f"frame_{frame_idx:04d}.png"), depth_vis)
            
            moge_results[f"frame_{frame_idx:04d}"] = {
                'depth': depth,
                'mask': mask,
                'intrinsics': intrinsics
            }
        
        # Save all results
        np.save(os.path.join(REPRO_DIR, "moge_results.npy"), moge_results)
        print(f"  ✓ Saved MoGe depth to: {MOGE_DEPTH_DIR}")
        print(f"  ✓ Saved MoGe visualizations to: {MOGE_VIS_DIR}")
        
    except ImportError as e:
        print(f"  ✗ Error: Could not import MoGe: {e}")
        return False
    
    return True


# ======================================================================
# STEP 3b: Load existing MoGe depth and save frame by frame
# ======================================================================
def step3b_load_existing_moge():
    """Load pre-computed MoGe depth from existing file and save frame by frame."""
    print("\n" + "="*60)
    print("STEP 3b: Loading Existing MoGe Depth")
    print("="*60)
    
    # Path to existing MoGe results (DNA_02 which is 0012_09)
    EXISTING_MOGE_PATH = "/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/data/DNA_Seq/depth_maps/DNA_02/DNA_Seqmoge_results.npy"
    
    if not os.path.exists(EXISTING_MOGE_PATH):
        print(f"  ✗ Error: Existing MoGe file not found at {EXISTING_MOGE_PATH}")
        return False
    
    os.makedirs(MOGE_DEPTH_DIR, exist_ok=True)
    os.makedirs(MOGE_VIS_DIR, exist_ok=True)
    
    print(f"  Loading from: {EXISTING_MOGE_PATH}")
    existing_data = np.load(EXISTING_MOGE_PATH, allow_pickle=True).item()
    existing_keys = sorted(list(existing_data.keys()))
    print(f"  Found {len(existing_keys)} frames in existing file")
    
    moge_results = {}
    
    for frame_idx in tqdm(range(NUM_FRAMES), desc="Processing MoGe data"):
        # Find matching key in existing data
        # Keys might be paths like "/path/to/frame_0000.png" or just "frame_0000"
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
        
        # Save depth as npy
        np.save(os.path.join(MOGE_DEPTH_DIR, f"frame_{frame_idx:04d}.npy"), depth)
        
        # Save visualization
        depth_vis = depth_to_color(depth)
        cv2.imwrite(os.path.join(MOGE_VIS_DIR, f"frame_{frame_idx:04d}.png"), depth_vis)
        
        moge_results[f"frame_{frame_idx:04d}"] = {
            'depth': depth,
            'mask': mask,
            'intrinsics': intrinsics
        }
    
    # Save all results in unified format
    np.save(os.path.join(REPRO_DIR, "moge_results.npy"), moge_results)
    print(f"  ✓ Saved {len(moge_results)} MoGe depth frames to: {MOGE_DEPTH_DIR}")
    print(f"  ✓ Saved MoGe visualizations to: {MOGE_VIS_DIR}")
    print(f"  ✓ Saved combined results to: {os.path.join(REPRO_DIR, 'moge_results.npy')}")
    
    return True


# ======================================================================
# STEP 4: Render PCL depth at raw resolution (1024x1224)
# ======================================================================
def step4_render_pcl_depth():
    """Render point cloud to depth maps at original resolution."""
    print("\n" + "="*60)
    print("STEP 4: Rendering PCL Depth at Raw Resolution (1024x1224)")
    print("="*60)
    
    os.makedirs(RENDERED_RAW_DIR, exist_ok=True)
    
    for frame_idx in tqdm(range(NUM_FRAMES), desc="Rendering PCL depth"):
        frame_dir = os.path.join(SEQ_DIR, str(frame_idx))
        
        # Load point cloud
        ply_path = os.path.join(frame_dir, "point_cloud", "iteration_20000", "point_cloud.ply")
        if not os.path.exists(ply_path):
            ply_path = os.path.join(frame_dir, "point_cloud", "iteration_15000", "point_cloud.ply")
        if not os.path.exists(ply_path):
            print(f"  Warning: No point cloud found for frame {frame_idx}")
            continue
        
        # Load cameras
        cameras_path = os.path.join(frame_dir, "cameras.json")
        cam = get_camera_for_view(cameras_path, VIEW_IDX)
        if cam is None:
            print(f"  Warning: Camera {VIEW_IDX} not found for frame {frame_idx}")
            continue
        
        # Load point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        points_world = np.asarray(pcd.points)
        
        # Get camera parameters
        width, height = cam['width'], cam['height']
        fx, fy = cam['fx'], cam['fy']
        cx, cy = width / 2.0, height / 2.0
        
        pos = np.array(cam['position'])
        rot = np.array(cam['rotation'])
        
        # Build extrinsic (world to camera)
        R_wc = rot
        R_cw = R_wc.T
        t_cw = -R_cw @ pos
        
        # Transform points to camera frame
        points_cam = (points_world @ R_cw.T) + t_cw
        
        # Filter points with Z > 0 (in front of camera)
        valid_z = points_cam[:, 2] > 0
        points_cam = points_cam[valid_z]
        
        if len(points_cam) == 0:
            continue
        
        # Project to image plane
        z = points_cam[:, 2]
        u = (points_cam[:, 0] * fx / z) + cx
        v = (points_cam[:, 1] * fy / z) + cy
        
        # Filter inside image bounds
        valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, z = u[valid_uv], v[valid_uv], z[valid_uv]
        
        if len(z) == 0:
            continue
        
        # Sort by depth (far to near) for proper occlusion
        sort_idx = np.argsort(z)[::-1]
        u_sorted, v_sorted, z_sorted = u[sort_idx], v[sort_idx], z[sort_idx]
        
        # Splat to depth map
        depth_map = np.zeros((height, width), dtype=np.float32)
        u_int = np.round(u_sorted).astype(int)
        v_int = np.round(v_sorted).astype(int)
        
        radius = 2
        for i in range(len(z_sorted)):
            cv2.circle(depth_map, (u_int[i], v_int[i]), radius, float(z_sorted[i]), -1)
        
        # Post-processing: fill small holes
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
        
        # Save
        np.save(os.path.join(RENDERED_RAW_DIR, f"frame_{frame_idx:04d}.npy"), depth_map)
        
        # Save visualization
        depth_vis = depth_to_color(depth_map)
        cv2.imwrite(os.path.join(RENDERED_RAW_DIR, f"frame_{frame_idx:04d}_vis.png"), depth_vis)
    
    print(f"  ✓ Saved raw rendered depth to: {RENDERED_RAW_DIR}")


# ======================================================================
# STEP 5: Center crop and resize rendered depth to 512x512
# ======================================================================
def step5_crop_rendered_depth():
    """Center crop and resize rendered depth to 512x512."""
    print("\n" + "="*60)
    print("STEP 5: Center Crop and Resize Rendered Depth to 512x512")
    print("="*60)
    
    os.makedirs(RENDERED_512_DIR, exist_ok=True)
    
    for frame_idx in tqdm(range(NUM_FRAMES), desc="Processing rendered depth"):
        src_path = os.path.join(RENDERED_RAW_DIR, f"frame_{frame_idx:04d}.npy")
        if not os.path.exists(src_path):
            continue
        
        depth = np.load(src_path)
        depth_cropped = centre_crop(depth, 1024)
        depth_resized = cv2.resize(depth_cropped, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        np.save(os.path.join(RENDERED_512_DIR, f"frame_{frame_idx:04d}.npy"), depth_resized)
        
        # Save visualization
        depth_vis = depth_to_color(depth_resized)
        cv2.imwrite(os.path.join(RENDERED_512_DIR, f"frame_{frame_idx:04d}_vis.png"), depth_vis)
    
    print(f"  ✓ Saved 512x512 rendered depth to: {RENDERED_512_DIR}")


# ======================================================================
# STEP 6: Calibrate MoGe depth using rendered depth
# ======================================================================
def step6_calibrate():
    """Calibrate MoGe depth using rendered GT depth."""
    print("\n" + "="*60)
    print("STEP 6: Calibrating MoGe Depth")
    print("="*60)
    
    os.makedirs(CALIBRATED_DIR, exist_ok=True)
    
    # Load MoGe results
    moge_path = os.path.join(REPRO_DIR, "moge_results.npy")
    if not os.path.exists(moge_path):
        print("  ✗ Error: MoGe results not found. Run step 3 first.")
        return None
    
    moge_data = np.load(moge_path, allow_pickle=True).item()
    
    scale_factors = []
    calibrated_depths = {}
    
    for frame_idx in tqdm(range(NUM_FRAMES), desc="Calibrating"):
        gt_path = os.path.join(RENDERED_512_DIR, f"frame_{frame_idx:04d}.npy")
        mask_path = os.path.join(MASKS_512_DIR, f"frame_{frame_idx:04d}.npy")
        moge_key = f"frame_{frame_idx:04d}"
        
        if not os.path.exists(gt_path) or moge_key not in moge_data:
            continue
        
        gt_depth = np.load(gt_path)
        moge_depth = moge_data[moge_key]['depth']
        moge_mask = moge_data[moge_key]['mask'].astype(bool)
        
        # Load foreground mask
        if os.path.exists(mask_path):
            fg_mask = np.load(mask_path)
        else:
            fg_mask = gt_depth > 0
        
        # Find overlap region
        overlap = fg_mask & moge_mask & (gt_depth > 0) & (moge_depth > 0)
        
        if overlap.sum() < 100:
            continue
        
        # Compute scale factor
        gt_median = np.median(gt_depth[overlap])
        moge_median = np.median(moge_depth[overlap])
        scale = gt_median / moge_median if moge_median > 0 else 1.0
        scale_factors.append(scale)
        
        # Apply calibration
        calibrated = moge_depth.copy() * scale
        
        calibrated_depths[frame_idx] = {
            'depth': calibrated,
            'mask': moge_mask,
            'fg_mask': fg_mask,
            'scale': scale,
            'intrinsics': moge_data[moge_key]['intrinsics']
        }
        
        # Save
        np.save(os.path.join(CALIBRATED_DIR, f"frame_{frame_idx:04d}.npy"), calibrated)
        
        # Save visualization
        depth_vis = depth_to_color(calibrated)
        cv2.imwrite(os.path.join(CALIBRATED_DIR, f"frame_{frame_idx:04d}_vis.png"), depth_vis)
    
    if scale_factors:
        print(f"\n  Scale factor statistics:")
        print(f"    Mean: {np.mean(scale_factors):.4f}")
        print(f"    Std:  {np.std(scale_factors):.4f}")
        print(f"    Range: [{np.min(scale_factors):.4f}, {np.max(scale_factors):.4f}]")
    
    print(f"  ✓ Saved calibrated depth to: {CALIBRATED_DIR}")
    return calibrated_depths


# ======================================================================
# STEP 7: Create visualizations
# ======================================================================
def step7_visualize(calibrated_depths):
    """Create side-by-side comparison visualizations."""
    print("\n" + "="*60)
    print("STEP 7: Creating Visualizations")
    print("="*60)
    
    os.makedirs(VIS_DIR, exist_ok=True)
    
    moge_data = np.load(os.path.join(REPRO_DIR, "moge_results.npy"), allow_pickle=True).item()
    
    for frame_idx in tqdm(list(calibrated_depths.keys()), desc="Creating visualizations"):
        # Load images
        rgb_path = os.path.join(FRAMES_512_DIR, f"frame_{frame_idx:04d}.png")
        if not os.path.exists(rgb_path):
            continue
        
        rgb = cv2.imread(rgb_path)
        gt_depth = np.load(os.path.join(RENDERED_512_DIR, f"frame_{frame_idx:04d}.npy"))
        moge_depth = moge_data[f"frame_{frame_idx:04d}"]['depth']
        calibrated = calibrated_depths[frame_idx]['depth']
        scale = calibrated_depths[frame_idx]['scale']
        
        # Common depth range for fair comparison
        gt_valid = gt_depth[gt_depth > 0]
        if gt_valid.size > 0:
            vmin, vmax = gt_valid.min(), gt_valid.max()
        else:
            vmin, vmax = 0, 1
        
        gt_vis = depth_to_color(gt_depth, vmin, vmax)
        moge_vis = depth_to_color(moge_depth)  # Original scale
        calibrated_vis = depth_to_color(calibrated, vmin, vmax)
        
        # Add labels
        cv2.putText(rgb, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(gt_vis, "GT Rendered", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(moge_vis, "MoGe (raw)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(calibrated_vis, f"Calibrated (s={scale:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        grid = np.hstack([rgb, gt_vis, moge_vis, calibrated_vis])
        cv2.imwrite(os.path.join(VIS_DIR, f"frame_{frame_idx:04d}.png"), grid)
    
    print(f"  ✓ Saved visualizations to: {VIS_DIR}")


# ======================================================================
# STEP 8: Create unprojected PLY comparisons
# ======================================================================
def step8_unproject_ply(calibrated_depths, frame_indices=None):
    """Create unprojected 3D point cloud comparisons."""
    print("\n" + "="*60)
    print("STEP 8: Creating Unprojected PLY Comparisons")
    print("="*60)
    
    os.makedirs(PLY_DIR, exist_ok=True)
    
    if frame_indices is None:
        frame_indices = [0, 30, 60, 90, 120]
    
    moge_data = np.load(os.path.join(REPRO_DIR, "moge_results.npy"), allow_pickle=True).item()
    
    for frame_idx in tqdm(frame_indices, desc="Unprojecting"):
        if frame_idx not in calibrated_depths:
            continue
        
        data = calibrated_depths[frame_idx]
        moge_key = f"frame_{frame_idx:04d}"
        
        # Get intrinsics
        moge_intr = data['intrinsics']
        H, W = 512, 512
        fx = moge_intr[0, 0] * W
        fy = moge_intr[1, 1] * H
        cx = moge_intr[0, 2] * W
        cy = moge_intr[1, 2] * H
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        # Load depths
        gt_depth = np.load(os.path.join(RENDERED_512_DIR, f"frame_{frame_idx:04d}.npy"))
        moge_depth = moge_data[moge_key]['depth']
        calibrated = data['depth']
        fg_mask = data['fg_mask']
        moge_mask = data['mask']
        bg_mask = moge_mask & ~fg_mask
        
        # Unproject
        gt_pts, _ = unproject_depth_to_3d(gt_depth, intrinsics, fg_mask)
        moge_fg_pts, _ = unproject_depth_to_3d(moge_depth, intrinsics, fg_mask)
        moge_bg_pts, _ = unproject_depth_to_3d(moge_depth, intrinsics, bg_mask)
        calib_fg_pts, _ = unproject_depth_to_3d(calibrated, intrinsics, fg_mask)
        calib_bg_pts, _ = unproject_depth_to_3d(calibrated, intrinsics, bg_mask)
        
        print(f"  Frame {frame_idx}: GT={len(gt_pts)}, MoGe_FG={len(moge_fg_pts)}, MoGe_BG={len(moge_bg_pts)}")
        
        # Create colored point clouds
        gt_pcd = create_colored_pointcloud(gt_pts, [0, 1, 0])  # Green
        moge_fg_pcd = create_colored_pointcloud(moge_fg_pts, [1, 0, 0])  # Red
        moge_bg_pcd = create_colored_pointcloud(moge_bg_pts, [1, 0.5, 0])  # Orange
        calib_fg_pcd = create_colored_pointcloud(calib_fg_pts, [0, 0.5, 1])  # Light blue
        calib_bg_pcd = create_colored_pointcloud(calib_bg_pts, [0, 0, 1])  # Blue
        
        # Save PLYs
        o3d.io.write_point_cloud(os.path.join(PLY_DIR, f"frame_{frame_idx:04d}_gt.ply"), gt_pcd)
        o3d.io.write_point_cloud(os.path.join(PLY_DIR, f"frame_{frame_idx:04d}_moge_fg.ply"), moge_fg_pcd)
        o3d.io.write_point_cloud(os.path.join(PLY_DIR, f"frame_{frame_idx:04d}_moge_bg.ply"), moge_bg_pcd)
        o3d.io.write_point_cloud(os.path.join(PLY_DIR, f"frame_{frame_idx:04d}_calibrated_fg.ply"), calib_fg_pcd)
        o3d.io.write_point_cloud(os.path.join(PLY_DIR, f"frame_{frame_idx:04d}_calibrated_bg.ply"), calib_bg_pcd)
        
        # Combined views
        o3d.io.write_point_cloud(os.path.join(PLY_DIR, f"frame_{frame_idx:04d}_gt_vs_moge.ply"), gt_pcd + moge_fg_pcd)
        o3d.io.write_point_cloud(os.path.join(PLY_DIR, f"frame_{frame_idx:04d}_gt_vs_calibrated.ply"), gt_pcd + calib_fg_pcd)
        o3d.io.write_point_cloud(os.path.join(PLY_DIR, f"frame_{frame_idx:04d}_full_scene.ply"), gt_pcd + calib_bg_pcd)
    
    print(f"  ✓ Saved PLY files to: {PLY_DIR}")
    print("  Color coding: Green=GT, Red=MoGe_FG, Orange=MoGe_BG, LightBlue=Calib_FG, Blue=Calib_BG")


# ======================================================================
# STEP 9: Create combined calibrated npy file
# ======================================================================
def step9_create_combined_npy():
    """Create combined moge_calibrated.npy file with all calibrated depths."""
    print("\n" + "="*60)
    print("STEP 9: Creating Combined Calibrated NPY File")
    print("="*60)
    
    MOGE_RESULTS = os.path.join(REPRO_DIR, "moge_results.npy")
    OUTPUT_PATH = os.path.join(REPRO_DIR, "moge_calibrated.npy")
    
    if not os.path.exists(MOGE_RESULTS):
        print(f"  ✗ Error: MoGe results not found at {MOGE_RESULTS}")
        return False
    
    # Load original MoGe results (to copy mask and intrinsics)
    moge_data = np.load(MOGE_RESULTS, allow_pickle=True).item()
    
    # Create calibrated output in same format
    calibrated_output = {}
    
    for frame_idx in tqdm(range(NUM_FRAMES), desc="Creating combined file"):
        moge_key = f"frame_{frame_idx:04d}"
        calib_path = os.path.join(CALIBRATED_DIR, f"frame_{frame_idx:04d}.npy")
        
        if not os.path.exists(calib_path) or moge_key not in moge_data:
            continue
        
        calibrated_depth = np.load(calib_path)
        
        calibrated_output[moge_key] = {
            'depth': calibrated_depth,
            'mask': moge_data[moge_key]['mask'],
            'intrinsics': moge_data[moge_key]['intrinsics']
        }
    
    # Save
    np.save(OUTPUT_PATH, calibrated_output)
    print(f"  ✓ Saved combined calibrated file to: {OUTPUT_PATH}")
    print(f"  ✓ Contains {len(calibrated_output)} frames")
    
    return True


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Process 0012_09 view 22 depth calibration")
    parser.add_argument('--step', type=str, default='all', help='Steps to run: all, or comma-separated (e.g., 1,2,3)')
    parser.add_argument('--frames', type=str, default='0,30,60,90,120', help='Frame indices for PLY export')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Processing {SEQ_NAME} view {VIEW_IDX}")
    print(f"Output: {REPRO_DIR}")
    print(f"{'='*60}")
    
    # Create all directories
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    
    # Parse steps
    if args.step.lower() == 'all':
        steps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        steps = [int(s.strip()) for s in args.step.split(',')]
    
    calibrated_depths = None
    
    if 1 in steps:
        step1_extract_frames()
    
    if 2 in steps:
        step2_crop_and_resize()
    
    if 3 in steps:
        step3_moge_depth()
    
    if 4 in steps:
        step4_render_pcl_depth()
    
    if 5 in steps:
        step5_crop_rendered_depth()
    
    if 6 in steps:
        calibrated_depths = step6_calibrate()
    
    if 7 in steps:
        if calibrated_depths is None:
            calibrated_depths = step6_calibrate()
        if calibrated_depths:
            step7_visualize(calibrated_depths)
    
    if 8 in steps:
        if calibrated_depths is None:
            calibrated_depths = step6_calibrate()
        if calibrated_depths:
            frame_indices = [int(x) for x in args.frames.split(',')]
            step8_unproject_ply(calibrated_depths, frame_indices)
    
    if 9 in steps:
        step9_create_combined_npy()
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"Output directory: {REPRO_DIR}")
    print(f"Folder structure:")
    print(f"  1_frames_raw/       - Raw extracted RGB frames")
    print(f"  1_masks_raw/        - Raw extracted masks")
    print(f"  2_frames_512/       - Center cropped + resized frames")
    print(f"  2_masks_512/        - Center cropped + resized masks")
    print(f"  3_moge_depth/       - MoGe depth estimation output")
    print(f"  3_moge_vis/         - MoGe depth visualization")
    print(f"  4_rendered_depth_raw/ - PCL depth at 1024x1224")
    print(f"  5_rendered_depth_512/ - PCL depth at 512x512")
    print(f"  6_calibrated_depth/ - Calibrated MoGe depth")
    print(f"  7_visualizations/   - Side-by-side comparisons")
    print(f"  8_unprojected_ply/  - 3D point cloud comparisons")
    print(f"  moge_calibrated.npy - Combined calibrated file (Step 9)")


if __name__ == "__main__":
    main()

