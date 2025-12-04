"""GPU-backed offline renderer using Open3D OffscreenRenderer

This script loads the same trajectory files as the interactive visualizer
(`pts3d1_p*.npy`, `pts3d2_p*.npy`) and renders frames using Open3D's
offscreen renderer (which can leverage GPU/OpenGL). It writes an MP4 via
`imageio`.

Notes:
- Requires `open3d` with the rendering module (pip package `open3d` >= 0.12
  typically provides `visualization.rendering.OffscreenRenderer`). If your
  Open3D build lacks offscreen support, the script will error and print
  instructions.
- This produces faster, GPU-accelerated output and will look different than
  the matplotlib output, but is much closer to real-time.

Usage example:
python offline_open3d_renderer.py --traj_path /path/to/traj_folder --output_dir /tmp/out --fps 20 --start_angle -30 --angle_range 60 --up_dir -z
"""

from pathlib import Path
import argparse
import numpy as np
from glob import glob
import imageio

def load_traj_dir(traj_path, use_float16=True):
    traj_path = Path(traj_path)
    traj_3d_head1 = None
    traj_3d_head2 = None

    traj_3d_paths_head1 = sorted(glob(str(traj_path / 'pts3d1_p*.npy')),
                                key=lambda x: int(x.split('_p')[-1].split('.')[0]))
    traj_3d_paths_head2 = sorted(glob(str(traj_path / 'pts3d2_p*.npy')),
                                key=lambda x: int(x.split('_p')[-1].split('.')[0]))

    if traj_3d_paths_head1:
        if use_float16:
            traj_3d_head1 = np.stack([np.load(p).astype(np.float16) for p in traj_3d_paths_head1], axis=0)
        else:
            traj_3d_head1 = np.stack([np.load(p) for p in traj_3d_paths_head1], axis=0)
        traj_3d_head1 = traj_3d_head1.reshape(traj_3d_head1.shape[0], -1, 6)

    if traj_3d_paths_head2:
        if use_float16:
            traj_3d_head2 = np.stack([np.load(p).astype(np.float16) for p in traj_3d_paths_head2], axis=0)
        else:
            traj_3d_head2 = np.stack([np.load(p) for p in traj_3d_paths_head2], axis=0)
        traj_3d_head2 = traj_3d_head2.reshape(traj_3d_head2.shape[0], -1, 6)

    return traj_3d_head1, traj_3d_head2


def center_and_prep(traj_3d_head1, traj_3d_head2, max_frames=None):
    center = None
    xyz_head1 = rgb_head1 = None
    xyz_head2 = rgb_head2 = None

    if traj_3d_head1 is not None:
        xyz_head1 = traj_3d_head1[:, :, :3].astype(np.float32)
        rgb_head1 = traj_3d_head1[:, :, 3:6].astype(np.float32)
        if center is None:
            center = np.mean(xyz_head1, axis=(0, 1), keepdims=True)
        xyz_head1 = xyz_head1 - center
        if rgb_head1.sum(axis=(-1)).max() > 125:
            rgb_head1 = rgb_head1 / 255.0

    if traj_3d_head2 is not None:
        xyz_head2 = traj_3d_head2[:, :, :3].astype(np.float32)
        rgb_head2 = traj_3d_head2[:, :, 3:6].astype(np.float32)
        if center is None and xyz_head2 is not None:
            center = np.mean(xyz_head2, axis=(0, 1), keepdims=True)
        if xyz_head2 is not None and center is not None:
            xyz_head2 = xyz_head2 - center
        if rgb_head2 is not None and rgb_head2.sum(axis=(-1)).max() > 125:
            rgb_head2 = rgb_head2 / 255.0

    F = max(
        xyz_head1.shape[0] if xyz_head1 is not None else 0,
        xyz_head2.shape[0] if xyz_head2 is not None else 0,
    ) if max_frames is None else max_frames

    if xyz_head1 is not None:
        xyz_head1 = xyz_head1[:F]
        rgb_head1 = rgb_head1[:F]
    if xyz_head2 is not None:
        xyz_head2 = xyz_head2[:F]
        rgb_head2 = rgb_head2[:F]

    return xyz_head1, rgb_head1, xyz_head2, rgb_head2


def make_open3d_renderer(width, height):
    try:
        import open3d as o3d
        from open3d.visualization import rendering
    except Exception as e:
        raise RuntimeError(
            "Open3D with rendering is required (pip install open3d).\n"
            "If Open3D is installed but lacks rendering support, install a build with GUI/renderer enabled."
        ) from e

    renderer = rendering.OffscreenRenderer(width, height)
    return o3d, renderer


def render_with_open3d(
    traj_path,
    output_path,
    fps=20,
    width=1280,
    height=720,
    downsample_factor=1,
    rotate=False,
    start_angle=0.0,
    angle_range=None,
    angle_per_frame=1.0,
    camera_pos=(1e-3, 0.75, -0.1),
    look_at=(0.0, 0.0, 0.0),
    up_dir='-z',
    max_frames=None,
    draw_trajectories=False,
    num_traj_points=100,
    # optional tinting parameters (from interactive visualizer camera JSON)
    blend_ratio=None,
    blue_rgb=None,
    red_rgb=None,
    point_size=None,
):
    o3d, renderer = make_open3d_renderer(width, height)

    traj_3d_head1, traj_3d_head2 = load_traj_dir(traj_path)
    xyz_head1, rgb_head1, xyz_head2, rgb_head2 = center_and_prep(traj_3d_head1, traj_3d_head2, max_frames)

    # Align up-axis convention with the matplotlib-based visualizer: when
    # the scene `up_dir` is '-z' (world up is negative Z) flip the z
    # coordinate so that the Open3D camera views front->back similarly.
    if up_dir == '-z':
        if xyz_head1 is not None:
            xyz_head1 = xyz_head1.copy()
            xyz_head1[..., 2] = -xyz_head1[..., 2]
        if xyz_head2 is not None:
            xyz_head2 = xyz_head2.copy()
            xyz_head2[..., 2] = -xyz_head2[..., 2]
        # flip camera and look_at z too
        camera_pos = (camera_pos[0], camera_pos[1], -camera_pos[2])
        look_at = (look_at[0], look_at[1], -look_at[2])

    # Auto-detect color channel order: if blue channel mean >> red channel
    # mean, assume BGR and swap to RGB. This helps when datasets store
    # colors in BGR (common from OpenCV pipelines).
    def maybe_swap_bgr_to_rgb(rgb_arr):
        if rgb_arr is None:
            return None
        # rgb_arr shape: (F, P, 3)
        mean_channels = rgb_arr.reshape(-1, 3).mean(axis=0)
        # If blue mean is significantly greater than red mean, swap
        if mean_channels[2] > mean_channels[0] * 1.2:
            rgb_arr = rgb_arr.copy()
            rgb_arr = rgb_arr[..., ::-1]
        return rgb_arr

    rgb_head1 = maybe_swap_bgr_to_rgb(rgb_head1)
    rgb_head2 = maybe_swap_bgr_to_rgb(rgb_head2)

    # handle up_dir
    up_map = {
        '+x': np.array([1.0, 0.0, 0.0]),
        '-x': np.array([-1.0, 0.0, 0.0]),
        '+y': np.array([0.0, 1.0, 0.0]),
        '-y': np.array([0.0, -1.0, 0.0]),
        '+z': np.array([0.0, 0.0, 1.0]),
        '-z': np.array([0.0, 0.0, -1.0]),
    }
    up_axis = up_map.get(up_dir, np.array([0.0, 0.0, -1.0]))

    # Build scene: add point clouds as geometry objects
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'

    # helper to add pointcloud to scene
    def add_pcd(name, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        renderer.scene.add_geometry(name, pcd, mat)

    # We'll re-create geometry per frame for simplicity (Open3D handles this fast on GPU)
    num_frames = max(
        xyz_head1.shape[0] if xyz_head1 is not None else 0,
        xyz_head2.shape[0] if xyz_head2 is not None else 0,
    )

    # Prefer legacy imageio writer for streaming/ffmpeg support; fall back
    # to imageio.v3 if needed.
    try:
        writer = imageio.get_writer(output_path, fps=fps)
    except Exception:
        try:
            import imageio.v3 as iio_v3
            writer = iio_v3.get_writer(output_path, fps=fps)
        except Exception as e:
            raise RuntimeError('No working imageio writer available') from e

    # Precompute angle sequence
    if angle_range is not None:
        denom = max(1, num_frames - 1)
        angles = [start_angle + (angle_range * i / denom) for i in range(num_frames)]
    else:
        angles = [start_angle + i * angle_per_frame for i in range(num_frames)]

    for i in range(num_frames):
        renderer.scene.clear_geometry()

        if xyz_head1 is not None and i < xyz_head1.shape[0]:
            pts = xyz_head1[i][::downsample_factor]
            cols = rgb_head1[i][::downsample_factor]
            # apply tinting if the interactive visualizer used it
            if (blend_ratio is not None) and (blue_rgb is not None):
                cols = cols * float(blend_ratio)
                cols = cols.copy()
                cols[:, 0] = np.clip(cols[:, 0] + float(blue_rgb[0]) * (1 - float(blend_ratio)), 0, 1)
                cols[:, 1] = np.clip(cols[:, 1] + float(blue_rgb[1]) * (1 - float(blend_ratio)), 0, 1)
                cols[:, 2] = np.clip(cols[:, 2] + float(blue_rgb[2]) * (1 - float(blend_ratio)), 0, 1)
            add_pcd('head1', pts, cols)

        if xyz_head2 is not None and i < xyz_head2.shape[0]:
            pts = xyz_head2[i][::downsample_factor]
            cols = rgb_head2[i][::downsample_factor]
            if (blend_ratio is not None) and (red_rgb is not None):
                cols = cols * float(blend_ratio)
                cols = cols.copy()
                cols[:, 0] = np.clip(cols[:, 0] + float(red_rgb[0]) * (1 - float(blend_ratio)), 0, 1)
                cols[:, 1] = np.clip(cols[:, 1] + float(red_rgb[1]) * (1 - float(blend_ratio)), 0, 1)
                cols[:, 2] = np.clip(cols[:, 2] + float(red_rgb[2]) * (1 - float(blend_ratio)), 0, 1)
            add_pcd('head2', pts, cols)

        # compute camera by rotating initial camera around up_axis
        cam_pos = np.array(camera_pos, dtype=float)
        look_at = np.array(look_at, dtype=float)
        rel = cam_pos - look_at
        # rotation using Rodrigues
        angle_rad = np.deg2rad(angles[i]) if rotate else np.deg2rad(angles[0])
        axis = up_axis / np.linalg.norm(up_axis)
        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)
        rel_rot = rel * cos_t + np.cross(axis, rel) * sin_t + axis * (np.dot(axis, rel)) * (1 - cos_t)
        cam_curr = look_at + rel_rot

        # Attempt to set camera; API differs across Open3D versions
        try:
            # look_at(center, eye, up)
            renderer.scene.camera.look_at(look_at, cam_curr, up_axis)
        except Exception:
            try:
                # setup_camera(vertical_fov, center, eye, up, ...)
                renderer.setup_camera(60.0, look_at, cam_curr, up_axis)
            except Exception:
                # Last resort: ignore and continue (camera will use default)
                pass

        img = renderer.render_to_image()
        # img is an Open3D Image; convert to numpy array
        try:
            arr = np.asarray(img)
        except Exception:
            # if render_to_image returns a PIL image-like, convert via buffer
            import PIL.Image as PILImage
            arr = np.array(PILImage.fromarray(img))

        # arr shape likely (H,W,3) or (H,W,4)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        writer.append_data(arr)

    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_path', required=True)
    parser.add_argument('--output_dir', default='vis_open3d_out')
    parser.add_argument('--camera_json', type=str, default=None, help='Optional path to camera_params.json produced by the interactive visualizer')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--max_frames', type=int, default=300)
    parser.add_argument('--downsample_factor', type=int, default=1)
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--start_angle', type=float, default=0.0)
    parser.add_argument('--angle_range', type=float, default=None)
    parser.add_argument('--angle_per_frame', type=float, default=1.0)
    parser.add_argument('--camera_pos', type=float, nargs=3, default=(1e-3, 0.75, -0.1))
    parser.add_argument('--look_at', type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument('--up_dir', type=str, default='-z')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--num_traj_points', type=int, default=100)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / (Path(args.traj_path).name + '.mp4')

    # If a camera JSON is provided, load overrides (position/look_at/up_dir,
    # tint settings) so the GPU renderer can match the interactive visualizer.
    cam_json = None
    if args.camera_json:
        try:
            import json
            cam_json = json.load(open(args.camera_json, 'r'))
        except Exception:
            cam_json = None

    cam_pos = tuple(args.camera_pos)
    look_at = tuple(args.look_at)
    up_dir = args.up_dir
    blend_ratio = None
    blue_rgb = None
    red_rgb = None
    point_size = None
    if cam_json is not None:
        cam_pos = tuple(cam_json.get('camera_position', cam_pos))
        look_at = tuple(cam_json.get('look_at', look_at))
        up_dir = cam_json.get('up_dir', up_dir)
        blend_ratio = cam_json.get('blend_ratio', None)
        blue_rgb = tuple(cam_json.get('blue_rgb')) if cam_json.get('blue_rgb') is not None else None
        red_rgb = tuple(cam_json.get('red_rgb')) if cam_json.get('red_rgb') is not None else None
        point_size = cam_json.get('point_size', None)
    render_with_open3d(
        args.traj_path,
        str(out_path),
        fps=args.fps,
        width=args.width,
        height=args.height,
        downsample_factor=args.downsample_factor,
        rotate=args.rotate,
        start_angle=args.start_angle,
        angle_range=args.angle_range,
        angle_per_frame=args.angle_per_frame,
        camera_pos=cam_pos,
        look_at=look_at,
        up_dir=up_dir,
        max_frames=args.max_frames,
        draw_trajectories=False,
        num_traj_points=args.num_traj_points,
        blend_ratio=blend_ratio,
        blue_rgb=blue_rgb,
        red_rgb=red_rgb,
        point_size=point_size,
    )


if __name__ == '__main__':
    main()
