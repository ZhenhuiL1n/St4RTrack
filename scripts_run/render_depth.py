import sys
# Add path to GS_render_script
sys.path.append('/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/datasets_preprocess/DNA_Rendering')

import argparse
import json
import os
import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import math

# Try importing GS modules
try:
    from GS_render_script import util_gau
    from GS_render_script.renderer_cuda import gaus_cuda_from_cpu
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    HAS_GS_RASTERIZER = True
except ImportError as e:
    print(f"Could not import GS_render_script: {e}")
    HAS_GS_RASTERIZER = False

def load_cameras(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_extrinsic_matrix(cam_data):
    # In standard Colmap/GS JSONs:
    # 'position' is the camera center C (in world coordinates).
    # 'rotation' is the rotation matrix R_wc (Camera -> World).
    # We need the World -> Camera transformation (Extrinsic Matrix).
    # P_cam = R_cw * (P_world - C)
    #       = R_cw * P_world - R_cw * C
    # So Extrinsic = [R_cw | -R_cw * C]
    # where R_cw = R_wc.T (inverse of rotation matrix)
    
    pos = np.array(cam_data['position'])
    rot = np.array(cam_data['rotation'])
    
    # Debug: Print determinant to check for reflection
    # print(f"Det: {np.linalg.det(rot)}")
    
    # Assuming rot is R_wc (Camera to World)
    R_wc = rot
    R_cw = R_wc.T
    t_cw = -R_cw @ pos
    
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_cw
    extrinsic[:3, 3] = t_cw
    
    return extrinsic, pos, R_wc

def create_camera_frustum(extrinsic, color=[1, 0, 0], scale=0.2):
    # Extrinsic is World-to-Camera (4x4)
    # We need Camera-to-World to draw the frustum
    cam_pose = np.linalg.inv(extrinsic)
    
    # Standard camera frustum in camera coordinates
    # Open3D camera looks down -Z, Y is up, X is right.
    
    points = [
        [0, 0, 0],
        [-1, -1, -2],
        [1, -1, -2],
        [1, 1, -2],
        [-1, 1, -2]
    ]
    points = np.array(points) * scale
    
    points_hom = np.hstack([points, np.ones((5, 1))])
    points_world = (cam_pose @ points_hom.T).T[:, :3]
    
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def render_depth(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cameras = load_cameras(args.cameras_json)
    
    cameras = load_cameras(args.cameras_json)
    
    import cv2
    
    # 1. Load Point Cloud
    print(f"Loading point cloud from {args.ply_path}...")
    pcd = o3d.io.read_point_cloud(args.ply_path)
    
    # NOTE: Do NOT center the point cloud! The extrinsics in cameras.json
    # are relative to the original point cloud position.
    # center = pcd.get_center()
    # pcd.translate(-center)
    
    print(f"Point Cloud Center: {pcd.get_center()} (NOT centering)")
    
    print(f"Rendering {len(cameras)} views (Manual Splatting)...")
    
    # Pre-compute centered points
    points_world = np.asarray(pcd.points)
    
    for cam in cameras:
        img_name = cam['img_name']
        width = cam['width']
        height = cam['height']
        # Use original focal lengths (no scaling)
        fx = cam['fx']
        fy = cam['fy']
        cx = width / 2.0
        cy = height / 2.0
        
        # Get Extrinsics (CV convention)
        extrinsic, pos, R_wc = get_extrinsic_matrix(cam)
        R_cw = extrinsic[:3, :3]
        t_cw = extrinsic[:3, 3]
        
        # 1. Transform to Camera Frame
        points_cam = (points_world @ R_cw.T) + t_cw
        
        # 2. Filter Z > 0
        valid_z = points_cam[:, 2] > 0
        points_cam = points_cam[valid_z]
        
        if len(points_cam) == 0:
            continue
            
        # 3. Project to Image Plane
        z = points_cam[:, 2]
        u = (points_cam[:, 0] * fx / z) + cx
        v = (points_cam[:, 1] * fy / z) + cy
        
        # 4. Filter inside image bounds
        valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u = u[valid_uv]
        v = v[valid_uv]
        z = z[valid_uv]
        
        if len(z) == 0:
            continue
            
        # 5. Sort by Depth (Far to Near)
        sort_idx = np.argsort(z)[::-1]
        u_sorted = u[sort_idx]
        v_sorted = v[sort_idx]
        z_sorted = z[sort_idx]
        
        # 6. Splat to Depth Image
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        u_int = np.round(u_sorted).astype(int)
        v_int = np.round(v_sorted).astype(int)
        
        radius = int(args.point_size)
        for i in range(len(z_sorted)):
            cv2.circle(depth_map, (u_int[i], v_int[i]), radius, float(z_sorted[i]), -1)
            
        # 7. Post-Processing: Densify and Smooth
        # Morphological Closing to fill small holes
        mask_valid = (depth_map > 0).astype(np.uint8)
        kernel_size = 5 # Adjust based on sparsity
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Close the mask to find where we SHOULD have depth
        mask_closed = cv2.morphologyEx(mask_valid, cv2.MORPH_CLOSE, kernel)
        
        # Fill holes
        # We can use simple dilation on depth, but that propagates max depth (background).
        # We want to propagate foreground.
        # Simple approximation: Median Blur handles holes well if they are small.
        
        # Apply Median Blur to smooth and fill
        # We run it on the raw depth map. 0s will be treated as values, which is bad.
        # So we only want to blur valid regions.
        
        # Better approach for "Denser":
        # 1. Inpaint the holes identified by (mask_closed - mask_valid).
        holes = (mask_closed - mask_valid).astype(np.uint8)
        if np.sum(holes) > 0:
            # cv2.inpaint expects 8-bit or 16-bit.
            # Convert to mm (uint16)
            depth_mm = (depth_map * 1000).astype(np.uint16)
            depth_inpainted = cv2.inpaint(depth_mm, holes, 3, cv2.INPAINT_TELEA)
            depth_map = depth_inpainted.astype(np.float32) / 1000.0
            
        # 2. Smooth with Median Blur (to reduce noise from splatting overlaps)
        depth_map = cv2.medianBlur(depth_map, 5)
        
        # Mask out the background again (outside the closed mask)
        depth_map = np.where(mask_closed > 0, depth_map, 0.0)

        # Save
        save_path = output_dir / f"{img_name}.npy"
        np.save(save_path, depth_map)
        
        if args.vis:
            plt.imsave(output_dir / f"{img_name}_depth.png", depth_map, cmap='plasma')
            
        # Reproject Verification (First Frame)
        if args.reproject and img_name == cameras[0]['img_name']:
             # Save Frustum PLY
             flip_yz = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
             extrinsic_gl = flip_yz @ extrinsic
             frustum = create_camera_frustum(extrinsic_gl, color=[1, 0, 0], scale=0.5)
             o3d.io.write_line_set(str(output_dir / f"{img_name}_frustum.ply"), frustum)
             print(f"Saved frustum visualization to {output_dir / f'{img_name}_frustum.ply'}")
             
             # Save Reprojection (Verification)
             visible_points_world = points_world[valid_z][valid_uv]
             pcd_vis = o3d.geometry.PointCloud()
             pcd_vis.points = o3d.utility.Vector3dVector(visible_points_world)
             o3d.io.write_point_cloud(str(output_dir / f"{img_name}_visible.ply"), pcd_vis)
             print(f"Saved visible points to {output_dir / f'{img_name}_visible.ply'}")
             
             # Real Reprojection (Depth Map -> 3D)
             print(f"Reprojecting {img_name} from DEPTH MAP for verification...")
             h, w = depth_map.shape
             u, v = np.meshgrid(np.arange(w), np.arange(h))
             z = depth_map
             
             valid_mask = (z > 0)
             
             x_cam = (u - cx) * z / fx
             y_cam = (v - cy) * z / fy
             z_cam = z
             
             pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)[valid_mask]
             
             #    to World
             pts_world_reproj = (pts_cam - t_cw) @ R_cw
             
             pcd_reproj = o3d.geometry.PointCloud()
             pcd_reproj.points = o3d.utility.Vector3dVector(pts_world_reproj)
             o3d.io.write_point_cloud(str(output_dir / f"{img_name}_reproj.ply"), pcd_reproj)
             print(f"Saved reprojection to {output_dir / f'{img_name}_reproj.ply'}")
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_path', required=True, help='Path to point_cloud.ply')
    parser.add_argument('--cameras_json', required=True, help='Path to cameras.json')
    parser.add_argument('--output_dir', required=True, help='Output directory for depth maps')
    parser.add_argument('--point_size', type=float, default=2.0, help='Point size for splatting')
    parser.add_argument('--vis', action='store_true', help='Save PNG visualizations')
    parser.add_argument('--reproject', action='store_true', help='Reproject first frame to 3D for verification')
    
    args = parser.parse_args()
    render_depth(args)
