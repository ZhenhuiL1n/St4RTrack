"""Offline visualizer for St4RTrack

Loads point-cloud trajectories (same format as `visualizer_st4rtrack.py`) and
saves an MP4 video rendering. This uses matplotlib's 3D scatter to render
frames offscreen and `imageio` to write the video.

Usage examples:
python offline_visualizer.py --traj_path results/calibrated_ari/checkpoints_.../ \
    --output_dir /tmp/vis_out --fps 20 --rotate --angle_per_frame 2.0

Batch mode (process all subfolders of a folder):
python offline_visualizer.py --traj_path results/calibrated_ari/ --output_dir /tmp/vis_out --batch

Notes:
- This is a lightweight, dependency-minimal offline renderer. For large
  point clouds or high frame counts consider using a GPU-based renderer.
"""

from pathlib import Path
import argparse
import numpy as np
from glob import glob
import imageio.v3 as iio
import json
import cv2
# Optional Open3D backend import. We import lazily where used but expose
# a friendly error if the module/file isn't present.
try:
    from offline_open3d_renderer import render_with_open3d  # type: ignore
except Exception:
    render_with_open3d = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm


def load_traj_dir(traj_path, use_float16=True):
    traj_path = Path(traj_path)
    traj_3d_head1 = None
    traj_3d_head2 = None

    # look for pts3d files same naming convention as visualizer_st4rtrack
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


def center_and_prep(traj_3d_head1, traj_3d_head2, num_frames=None):
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

    if num_frames is None:
        F = max(
            xyz_head1.shape[0] if xyz_head1 is not None else 0,
            xyz_head2.shape[0] if xyz_head2 is not None else 0,
        )
    else:
        F = num_frames

    if xyz_head1 is not None:
        xyz_head1 = xyz_head1[:F]
        rgb_head1 = rgb_head1[:F]
    if xyz_head2 is not None:
        xyz_head2 = xyz_head2[:F]
        rgb_head2 = rgb_head2[:F]

    return xyz_head1, rgb_head1, xyz_head2, rgb_head2


def render_to_video(
    xyz_head1,
    rgb_head1,
    xyz_head2,
    rgb_head2,
    output_path,
    fps=20,
    figsize=(1280, 720),
    downsample_factor=1,
    rotate=False,
    angle_per_frame=1.0,
    start_angle=0.0,
    angle_range=None,
    elev=20,
    point_size=1.0,
    camera_pos=(1e-3, 0.75, -0.1),
    look_at=(0.0, 0.0, 0.0),
    draw_trajectories=False,
    num_traj_points=100,
    traj_start_frame=0,
    traj_end_frame=None,
    fixed_length_traj=10,
    traj_line_width=2.0,
    color_code='jet',
    conf_mask_head1=None,
    conf_mask_head2=None,
    use_color_tint=True,
    blend_ratio=0.7,
    blue_rgb=(0.0, 0.149, 0.463),
    red_rgb=(0.769, 0.510, 0.055),
    bbox_scale=1.0,
):
    # Determine number of frames
    num_frames = max(
        xyz_head1.shape[0] if xyz_head1 is not None else 0,
        xyz_head2.shape[0] if xyz_head2 is not None else 0,
    )

    # Set up global limits for stable view
    all_xyz = []
    if xyz_head1 is not None:
        all_xyz.append(xyz_head1.reshape(-1, 3))
    if xyz_head2 is not None:
        all_xyz.append(xyz_head2.reshape(-1, 3))
    if all_xyz:
        all_xyz = np.concatenate(all_xyz, axis=0)
        mins = all_xyz.min(axis=0)
        maxs = all_xyz.max(axis=0)
    else:
        mins = np.array([-1, -1, -1])
        maxs = np.array([1, 1, 1])

    xlim = (mins[0], maxs[0])
    ylim = (mins[1], maxs[1])
    zlim = (mins[2], maxs[2])

    # Optionally shrink/expand the bounding box around the center to crop out
    # empty space (bbox_scale < 1.0 will zoom/crop tighter)
    try:
        if bbox_scale is not None and float(bbox_scale) != 1.0:
            center = (mins + maxs) / 2.0
            half = (maxs - mins) / 2.0
            half = half * float(bbox_scale)
            mins2 = center - half
            maxs2 = center + half
            xlim = (mins2[0], maxs2[0])
            ylim = (mins2[1], maxs2[1])
            zlim = (mins2[2], maxs2[2])
    except Exception:
        pass

    # imageio v3 moved APIs; fall back to legacy imageio.get_writer if needed
    try:
        writer = iio.get_writer(output_path, fps=fps)
    except Exception:
        import imageio as _imageio_legacy

        # prefer imageio-ffmpeg if available for mp4 writing
        writer = _imageio_legacy.get_writer(output_path, fps=fps)

    fig = plt.figure(figsize=(figsize[0] / 100, figsize[1] / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Prepare trajectories if requested (based on head1)
    trajs_selected = None
    if draw_trajectories and xyz_head1 is not None:
        t_end = traj_end_frame if traj_end_frame is not None else num_frames
        t_end = min(t_end, num_frames)
        trajs = xyz_head1[traj_start_frame:t_end]  # (T, P, 3)
        if trajs.shape[0] > 1:
            traj_diffs = np.diff(trajs, axis=0)
            traj_lengths = np.sum(np.sqrt(np.sum(traj_diffs ** 2, axis=-1)), axis=0)
            # pick top num_traj_points by length
            idx = np.argsort(-traj_lengths)
            pick = idx[:min(num_traj_points, len(idx))]
            trajs_selected = trajs[:, pick, :]
        else:
            trajs_selected = None

    # compute initial azim/elev from camera_pos
    cam_pos = np.array(camera_pos, dtype=float)
    look_at = np.array(look_at, dtype=float)
    # If the render_to_video.up_axis was set by caller (from args.up_dir),
    # transform coordinates into matplotlib's default +z up convention when
    # necessary. The interactive visualizer uses `-z` by default in this
    # project, so if up_axis == -z we flip the z coordinate for points and
    # camera to make the matplotlib view consistent with the web viewer.
    try:
        up_axis = render_to_video.up_axis
    except AttributeError:
        up_axis = np.array([0.0, 0.0, -1.0])

    # If up_axis is -z (i.e., scene up is negative Z), flip z for plotting
    if np.allclose(up_axis, np.array([0.0, 0.0, -1.0])):
        if xyz_head1 is not None:
            xyz_head1 = xyz_head1.copy()
            xyz_head1[..., 2] = -xyz_head1[..., 2]
        if xyz_head2 is not None:
            xyz_head2 = xyz_head2.copy()
            xyz_head2[..., 2] = -xyz_head2[..., 2]
        cam_pos[2] = -cam_pos[2]
        look_at[2] = -look_at[2]
    rel = cam_pos - look_at
    rho = np.linalg.norm(rel)
    # avoid division by zero
    xy = np.linalg.norm(rel[:2])
    init_elev = float(np.degrees(np.arctan2(rel[2], xy))) if rho > 0 else float(elev)
    init_azim = float(np.degrees(np.arctan2(rel[1], rel[0])))

    # Prepare up-axis for orbiting. Default is z-up negative (matches visualizer '-z').
    def up_dir_to_vec(up_dir_str: str):
        s = up_dir_str.strip()
        if s == '+x':
            return np.array([1.0, 0.0, 0.0])
        if s == '-x':
            return np.array([-1.0, 0.0, 0.0])
        if s == '+y':
            return np.array([0.0, 1.0, 0.0])
        if s == '-y':
            return np.array([0.0, -1.0, 0.0])
        if s == '+z':
            return np.array([0.0, 0.0, 1.0])
        if s == '-z':
            return np.array([0.0, 0.0, -1.0])
        # fallback
        return np.array([0.0, 0.0, 1.0])

    # Rodrigues rotation helper
    def rotate_vec_around_axis(vec, axis, angle_rad):
        axis = axis / np.linalg.norm(axis)
        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)
        return vec * cos_t + np.cross(axis, vec) * sin_t + axis * (np.dot(axis, vec)) * (1 - cos_t)

    # Determine orbit up axis (string provided by caller in args via process_single)
    orbit_up_vec = None
    if isinstance(look_at, tuple) or isinstance(look_at, list):
        pass

    # Create artists once to speed up rendering: reuse scatter and Line3D objects
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_axis_off()

    head1_artist = None
    head2_artist = None
    traj_line_objs = []

    # initialize head1 and head2 artists with the first available frame
    if xyz_head1 is not None and xyz_head1.shape[0] > 0:
        p0 = xyz_head1[0]
        c0 = rgb_head1[0]
        # apply confidence mask for first frame if provided
        if conf_mask_head1 is not None:
            try:
                mask0 = conf_mask_head1[0]
                p0 = p0[mask0]
                c0 = c0[mask0]
            except Exception:
                pass
        p0 = p0[::downsample_factor]
        c0 = c0[::downsample_factor]
        head1_artist = ax.scatter(p0[:, 0], p0[:, 1], p0[:, 2], c=c0, s=point_size, marker='.', depthshade=False)

    if xyz_head2 is not None and xyz_head2.shape[0] > 0:
        p0 = xyz_head2[0]
        c0 = rgb_head2[0]
        if conf_mask_head2 is not None:
            try:
                mask0 = conf_mask_head2[0]
                p0 = p0[mask0]
                c0 = c0[mask0]
            except Exception:
                pass
        p0 = p0[::downsample_factor]
        c0 = c0[::downsample_factor]
        head2_artist = ax.scatter(p0[:, 0], p0[:, 1], p0[:, 2], c=c0, s=point_size, marker='.', depthshade=False)

    # Prepare trajectory line objects (empty) if requested
    if trajs_selected is not None:
        T, N, _ = trajs_selected.shape
        for p in range(N):
            # create empty line, will set data each frame
            line_obj, = ax.plot([], [], [], linewidth=traj_line_width, alpha=0.9)
            traj_line_objs.append(line_obj)

    for i in tqdm(range(num_frames), desc=f"Rendering -> {Path(output_path).name}"):
        # Update point clouds
        if xyz_head1 is not None and i < xyz_head1.shape[0] and head1_artist is not None:
            pos = xyz_head1[i]
            col = rgb_head1[i]
            if conf_mask_head1 is not None:
                try:
                    pos = pos[conf_mask_head1[i]]
                    col = col[conf_mask_head1[i]]
                except Exception:
                    pass
            pos = pos[::downsample_factor]
            col = col[::downsample_factor]
            # update offsets (for 3D scatter use _offsets3d)
            try:
                head1_artist._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
            except Exception:
                # fallback: remove and re-create if update not supported
                head1_artist.remove()
                head1_artist = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=col, s=point_size, marker='.', depthshade=False)
            try:
                head1_artist.set_facecolor(col)
            except Exception:
                pass

        if xyz_head2 is not None and i < xyz_head2.shape[0] and head2_artist is not None:
            pos = xyz_head2[i]
            col = rgb_head2[i]
            if conf_mask_head2 is not None:
                try:
                    pos = pos[conf_mask_head2[i]]
                    col = col[conf_mask_head2[i]]
                except Exception:
                    pass
            pos = pos[::downsample_factor]
            col = col[::downsample_factor]
            try:
                head2_artist._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
            except Exception:
                head2_artist.remove()
                head2_artist = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=col, s=point_size, marker='.', depthshade=False)
            try:
                head2_artist.set_facecolor(col)
            except Exception:
                pass

        # Update trajectory lines up to current frame
        if trajs_selected is not None and len(traj_line_objs) > 0:
            T = trajs_selected.shape[0]
            t_show = min(i - traj_start_frame + 1, T) if i >= traj_start_frame else 0
            if t_show > 1:
                for p, line_obj in enumerate(traj_line_objs):
                    seg = trajs_selected[:t_show, p, :]
                    line_obj.set_data(seg[:, 0], seg[:, 1])
                    # set z data for 3D line
                    try:
                        line_obj.set_3d_properties(seg[:, 2])
                    except Exception:
                        pass

        # If rotating, compute new camera position by rotating the initial rel vector
        if rotate:
            # compute orbit axis (use up_dir if provided in closure scope)
            try:
                up_axis = render_to_video.up_axis
            except AttributeError:
                # default to -z to match the interactive visualizer
                up_axis = np.array([0.0, 0.0, -1.0])

            # compute per-frame rotation angle. Support explicit start+range or per-frame increment
            if angle_range is not None:
                # distribute the angle_range from start_angle across frames
                denom = max(1, num_frames - 1)
                angle_deg = start_angle + (angle_range * i / denom)
            else:
                angle_deg = start_angle + i * angle_per_frame

            angle_rad = np.deg2rad(angle_deg)
            rel_rot = rotate_vec_around_axis(rel, up_axis, angle_rad)
            cam_rot = look_at + rel_rot
            # compute azim/elev from rotated camera vector
            rel2 = cam_rot - look_at
            rho2 = np.linalg.norm(rel2)
            xy2 = np.linalg.norm(rel2[:2])
            elev_curr = float(np.degrees(np.arctan2(rel2[2], xy2))) if rho2 > 0 else float(elev)
            azim = float(np.degrees(np.arctan2(rel2[1], rel2[0])))
            # apply optional azimuth offset and debug/force overrides
            azim += getattr(render_to_video, 'azim_offset', 0.0)
            if getattr(render_to_video, 'force_azim', None) is not None:
                azim = render_to_video.force_azim
            if getattr(render_to_video, 'force_elev', None) is not None:
                elev_curr = render_to_video.force_elev
            if getattr(render_to_video, 'debug', False) and i < 5:
                print(f"[debug] frame={i} rel2={rel2.tolist()} azim={azim:.2f} elev={elev_curr:.2f}")
        else:
            azim = init_azim
            elev_curr = init_elev

        ax.view_init(elev=elev_curr, azim=azim)

        fig.canvas.draw()
        # Grab the pixel buffer and append to video. Different matplotlib
        # versions expose different canvas buffer methods; handle both.
        w, h = fig.canvas.get_width_height()
        try:
            buf = fig.canvas.tostring_rgb()
            img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
        except AttributeError:
            # Older/newer matplotlib may provide ARGB buffer
            try:
                buf = fig.canvas.tostring_argb()
            except AttributeError:
                # Last-resort: use canvas.buffer_rgba() if available
                try:
                    buf = fig.canvas.buffer_rgba()
                    # buffer_rgba may return an object with 'to_string' or similar
                    if hasattr(buf, 'tobytes'):
                        arr = np.frombuffer(buf.tobytes(), dtype=np.uint8)
                    else:
                        arr = np.frombuffer(bytes(buf), dtype=np.uint8)
                    arr = arr.reshape((h, w, 4))
                    # ARGB -> RGB
                    img = arr[:, :, [1, 2, 3]]
                except Exception:
                    raise
            else:
                arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
                # arr is ARGB (A,R,G,B) -> convert to RGB
                img = arr[:, :, [1, 2, 3]]

        writer.append_data(img)

    writer.close()
    plt.close(fig)


def process_single(traj_path, output_dir, args):
    traj_path = Path(traj_path)
    if not traj_path.exists():
        print(f"Path does not exist: {traj_path}")
        return

    # Attempt to read camera params exported by the online visualizer unless the
    # user explicitly asked to ignore them via --ignore_camera_params.
    cam_params_file = traj_path / 'camera_params.json'
    if cam_params_file.exists():
        if getattr(args, 'ignore_camera_params', False):
            print(f"Skipping camera params in {cam_params_file} because --ignore_camera_params was set")
        else:
            try:
                with open(cam_params_file, 'r') as f:
                    cam = json.load(f)
                # Apply camera parameters if present
                if 'camera_position' in cam:
                    args.camera_pos = tuple(cam['camera_position'])
                if 'look_at' in cam:
                    args.look_at = tuple(cam['look_at'])
                if 'up_dir' in cam:
                    args.up_dir = cam['up_dir']
                if 'point_size' in cam:
                    args.point_size = float(cam['point_size'])
                if 'blend_ratio' in cam:
                    args.blend_ratio = float(cam.get('blend_ratio', args.blend_ratio))
                if 'blue_rgb' in cam:
                    args.blue_rgb = tuple(cam.get('blue_rgb', args.blue_rgb))
                if 'red_rgb' in cam:
                    args.red_rgb = tuple(cam.get('red_rgb', args.red_rgb))
                print(f"Loaded camera params from {cam_params_file}")
            except Exception:
                pass

    # Apply zoom factor to camera_pos if requested (s<1 zooms in)
    try:
        if getattr(args, 'zoom', 1.0) is not None and float(args.zoom) != 1.0:
            la = tuple(getattr(args, 'look_at', (0.0, 0.0, 0.0)))
            cp = tuple(getattr(args, 'camera_pos', (1e-3, 0.75, -0.1)))
            s = float(args.zoom)
            new_cp = tuple(la[i] + s * (cp[i] - la[i]) for i in range(3))
            args.camera_pos = new_cp
            print(f"Applied zoom {s}: new camera_pos={args.camera_pos}")
    except Exception:
        pass

    # Print out the final camera parameters so it's clear what will be used for rendering
    try:
        print(f"Using camera_pos={tuple(getattr(args, 'camera_pos', None))} look_at={tuple(getattr(args, 'look_at', None))} up_dir={getattr(args, 'up_dir', None)} point_size={getattr(args, 'point_size', None)}")
    except Exception:
        pass

    # If it's a directory, load pts3d files
    conf_mask_head1 = None
    conf_mask_head2 = None

    if traj_path.is_dir():
        traj_3d_head1, traj_3d_head2 = load_traj_dir(traj_path, use_float16=not args.no_float16)

        # load confidence files if present and compute masks similar to interactive visualizer
        traj_3d_paths_head1 = sorted(glob(str(traj_path / 'pts3d1_p*.npy')),
                                    key=lambda x: int(x.split('_p')[-1].split('.')[0]))
        traj_3d_paths_head2 = sorted(glob(str(traj_path / 'pts3d2_p*.npy')),
                                    key=lambda x: int(x.split('_p')[-1].split('.')[0]))

        conf_paths_head1 = sorted(glob(str(traj_path / 'conf1_p*.npy')),
                                  key=lambda x: int(x.split('_p')[-1].split('.')[0]))
        conf_paths_head2 = sorted(glob(str(traj_path / 'conf2_p*.npy')),
                                  key=lambda x: int(x.split('_p')[-1].split('.')[0]))

        # Process confidence for head1
        if traj_3d_head1 is not None and conf_paths_head1:
            try:
                conf_head1 = np.stack([np.load(p).astype(np.float16) for p in conf_paths_head1], axis=0)
                conf_head1 = conf_head1.reshape(conf_head1.shape[0], -1)
                # average confidences across frames like online viewer
                conf_mean = conf_head1.mean(axis=0)
                conf_tiled = np.tile(conf_mean, (traj_3d_head1.shape[0], 1))
                conf_thre = np.percentile(conf_tiled.astype(np.float32), args.conf_thre_percentile)
                conf_mask_head1 = conf_tiled > conf_thre
            except Exception:
                conf_mask_head1 = None

        # Process confidence for head2
        if traj_3d_head2 is not None and conf_paths_head2:
            try:
                conf_head2 = np.stack([np.load(p).astype(np.float16) for p in conf_paths_head2], axis=0)
                conf_head2 = conf_head2.reshape(conf_head2.shape[0], -1)
                # compute per-frame percentile threshold like online viewer
                conf_thre = np.percentile(conf_head2.astype(np.float32), args.conf_thre_percentile, axis=1)
                # broadcast to create mask per frame
                conf_mask_head2 = conf_head2 > conf_thre[:, None]
            except Exception:
                conf_mask_head2 = None
    else:
        # Not a directory: attempt to load as numpy archive or single file
        if traj_path.suffix in ['.npz', '.npy']:
            arr = np.load(traj_path)
            # Expect arrays named like pts3d1 or similar
            if 'pts3d1' in arr:
                traj_3d_head1 = arr['pts3d1']
            else:
                traj_3d_head1 = None
            traj_3d_head2 = None
        else:
            print(f"Unsupported file type for: {traj_path}. Expect a directory or .npz/.npy archive.")
            return

    xyz_head1, rgb_head1, xyz_head2, rgb_head2 = center_and_prep(traj_3d_head1, traj_3d_head2, num_frames=args.max_frames)

    # Apply color tinting to match online visualizer (if enabled)
    # The interactive visualizer adds a frame with a 90deg rotation around X
    # (wxyz=tf.SO3.exp([pi/2,0,0])). To reproduce the exact orientation we
    # rotate the point clouds by the same matrix before rendering.
    try:
        rmat = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        if xyz_head1 is not None:
            xyz_head1 = xyz_head1 @ rmat.T
        if xyz_head2 is not None:
            xyz_head2 = xyz_head2 @ rmat.T
    except Exception:
        pass

    if getattr(args, 'use_color_tint', True):
        if rgb_head1 is not None:
            try:
                c = rgb_head1.copy().astype(np.float32)
                c *= float(getattr(args, 'blend_ratio', 0.7))
                br = tuple(getattr(args, 'blue_rgb', (0.0, 0.149, 0.463)))
                c[..., 0] = np.clip(c[..., 0] + br[0] * (1 - float(getattr(args, 'blend_ratio', 0.7))), 0, 1)
                c[..., 1] = np.clip(c[..., 1] + br[1] * (1 - float(getattr(args, 'blend_ratio', 0.7))), 0, 1)
                c[..., 2] = np.clip(c[..., 2] + br[2] * (1 - float(getattr(args, 'blend_ratio', 0.7))), 0, 1)
                rgb_head1 = c
            except Exception:
                pass

        if rgb_head2 is not None:
            try:
                c = rgb_head2.copy().astype(np.float32)
                c *= float(getattr(args, 'blend_ratio', 0.7))
                rr = tuple(getattr(args, 'red_rgb', (0.769, 0.510, 0.055)))
                c[..., 0] = np.clip(c[..., 0] + rr[0] * (1 - float(getattr(args, 'blend_ratio', 0.7))), 0, 1)
                c[..., 1] = np.clip(c[..., 1] + rr[1] * (1 - float(getattr(args, 'blend_ratio', 0.7))), 0, 1)
                c[..., 2] = np.clip(c[..., 2] + rr[2] * (1 - float(getattr(args, 'blend_ratio', 0.7))), 0, 1)
                rgb_head2 = c
            except Exception:
                pass

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output path
    out_name = traj_path.name.rstrip('/') + '.mp4'
    output_path = output_dir / out_name

    # If user requested the Open3D backend, delegate to that renderer which
    # loads the same `pts3d` files internally. This keeps the Open3D path
    # independent and leverages GPU/EGL support.
    if getattr(args, 'backend', 'matplotlib') == 'open3d':
        if render_with_open3d is None:
            print("Open3D backend requested but `offline_open3d_renderer.render_with_open3d` is not available.")
            print("Ensure `offline_open3d_renderer.py` is present and Open3D is installed in your environment.")
            return

        # call Open3D renderer with matching parameters
        try:
            render_with_open3d(
                str(traj_path),
                str(output_path),
                fps=args.fps,
                width=args.width,
                height=args.height,
                downsample_factor=args.downsample_factor,
                rotate=args.rotate,
                start_angle=getattr(args, 'start_angle', 0.0),
                angle_range=getattr(args, 'angle_range', None),
                angle_per_frame=args.angle_per_frame,
                camera_pos=tuple(args.camera_pos),
                look_at=tuple(args.look_at),
                up_dir=getattr(args, 'up_dir', '-z'),
                max_frames=args.max_frames,
                draw_trajectories=args.draw_trajectories,
                num_traj_points=args.num_traj_points,
            )
        except Exception as e:
            print('Open3D rendering failed:', e)
        return

    # set orbit up axis for render loop (map from args.up_dir string)
    up_map = {
        '+x': np.array([1.0, 0.0, 0.0]),
        '-x': np.array([-1.0, 0.0, 0.0]),
        '+y': np.array([0.0, 1.0, 0.0]),
        '-y': np.array([0.0, -1.0, 0.0]),
        '+z': np.array([0.0, 0.0, 1.0]),
        '-z': np.array([0.0, 0.0, -1.0]),
    }
    up_vec = up_map.get(getattr(args, 'up_dir', '-z'), np.array([0.0, 0.0, -1.0]))
    render_to_video.up_axis = up_vec

    # set debug/force parameters on the function so render loop can read them
    render_to_video.azim_offset = args.azim_offset
    render_to_video.force_azim = args.force_azim
    render_to_video.force_elev = args.force_elev
    render_to_video.debug = args.debug

    render_to_video(
        xyz_head1,
        rgb_head1,
        xyz_head2,
        rgb_head2,
        str(output_path),
        fps=args.fps,
        figsize=(args.width, args.height),
        downsample_factor=args.downsample_factor,
        rotate=args.rotate,
        angle_per_frame=args.angle_per_frame,
        start_angle=getattr(args, 'start_angle', 0.0),
        angle_range=getattr(args, 'angle_range', None),
        elev=args.elev,
        point_size=args.point_size,
        camera_pos=tuple(args.camera_pos),
        look_at=tuple(args.look_at),
        conf_mask_head1=conf_mask_head1,
        conf_mask_head2=conf_mask_head2,
        use_color_tint=getattr(args, 'use_color_tint', True),
        blend_ratio=getattr(args, 'blend_ratio', 0.7),
        blue_rgb=getattr(args, 'blue_rgb', (0.0, 0.149, 0.463)),
        red_rgb=getattr(args, 'red_rgb', (0.769, 0.510, 0.055)),
        bbox_scale=getattr(args, 'bbox_scale', 1.0),
        draw_trajectories=args.draw_trajectories,
        num_traj_points=args.num_traj_points,
        traj_start_frame=args.traj_start_frame,
        traj_end_frame=args.traj_end_frame,
        traj_line_width=args.traj_line_width,
        color_code=args.color_code,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_path', required=True, help='Path to trajectory folder or a parent folder in batch mode')
    parser.add_argument('--output_dir', default='vis_offline_out', help='Directory to write MP4 outputs')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--max_frames', type=int, default=300)
    parser.add_argument('--downsample_factor', type=int, default=1)
    parser.add_argument('--zoom', type=float, default=1.0, help='Zoom factor applied to camera distance from look_at (s<1 zooms in, s>1 zooms out)')
    parser.add_argument('--bbox_scale', type=float, default=1.0, help='Scale applied to the scene bounding box (0..1 shrinks crop, >1 expands)')
    parser.add_argument('--ignore_camera_params', action='store_true', help='If set, do not load camera_params.json from the trajectory folder (use CLI camera_pos/look_at instead)')
    parser.add_argument('--rotate', action='store_true', help='Rotate camera across frames')
    parser.add_argument('--angle_per_frame', type=float, default=1.0, help='Degrees to rotate per frame when --rotate')
    parser.add_argument('--elev', type=float, default=20.0, help='Elevation angle for camera')
    parser.add_argument('--start_angle', type=float, default=0.0, help='Start angle (degrees) for orbiting')
    parser.add_argument('--angle_range', type=float, default=None, help='Total angle range (degrees) to sweep across the video; if set, overrides --angle_per_frame distribution')
    parser.add_argument('--camera_pos', type=float, nargs=3, default=(1e-3, 0.35, -0.1), help='Initial camera position x y z')
    parser.add_argument('--look_at', type=float, nargs=3, default=(0.0, 0.0, 0.0), help='Point to look at (x y z)')
    parser.add_argument('--draw_trajectories', action='store_true', help='Draw trajectories as lines')
    parser.add_argument('--num_traj_points', type=int, default=100, help='Number of trajectory points to show')
    parser.add_argument('--traj_start_frame', type=int, default=0, help='Trajectory start frame')
    parser.add_argument('--traj_end_frame', type=int, default=None, help='Trajectory end frame')
    parser.add_argument('--traj_line_width', type=float, default=2.0, help='Line width for trajectories')
    parser.add_argument('--color_code', type=str, default='jet', help='Color map for trajectories (jet/rainbow)')
    parser.add_argument('--up_dir', type=str, default='-z', help='Up direction for orbiting: +x,-x,+y,-y,+z,-z')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--point_size', type=float, default=1.0)
    parser.add_argument('--backend', type=str, default='open3d', choices=['matplotlib', 'open3d'], help='Rendering backend to use')
    parser.set_defaults(use_color_tint=True)
    parser.add_argument('--no_color_tint', dest='use_color_tint', action='store_false', help='Disable color tinting to match web viewer')
    parser.add_argument('--blend_ratio', type=float, default=0.7, help='Blend ratio for color tinting (0..1)')
    parser.add_argument('--blue_rgb', type=float, nargs=3, default=(0.0, 0.149, 0.463), help='Blue tint RGB for head1')
    parser.add_argument('--red_rgb', type=float, nargs=3, default=(0.769, 0.510, 0.055), help='Red tint RGB for head2')
    parser.add_argument('--conf_thre_percentile', type=float, default=0.0, help='Percentile for confidence thresholding (0..100)')
    parser.add_argument('--force_azim', type=float, default=None, help='Force the camera azimuth (deg) for all frames')
    parser.add_argument('--force_elev', type=float, default=None, help='Force the camera elevation (deg) for all frames')
    parser.add_argument('--azim_offset', type=float, default=0.0, help='Additive offset (deg) to computed azimuth')
    parser.add_argument('--debug', action='store_true', help='Print debug info about camera angles for first frames')
    parser.add_argument('--batch', action='store_true', help='If given and traj_path is a directory, process all subdirectories')
    parser.add_argument('--no-float16', dest='no_float16', action='store_true', help='Do not cast loaded arrays to float16')
    # restore default backend to matplotlib for safe CPU rendering
    parser.set_defaults(backend='matplotlib')
    args = parser.parse_args()

    traj_path = Path(args.traj_path)
    output_dir = Path(args.output_dir)

    if args.batch and traj_path.is_dir():
        subdirs = [p for p in sorted(traj_path.iterdir()) if p.is_dir()]
        if not subdirs:
            # maybe the directory itself is a trajectory folder
            subdirs = [traj_path]
        for sub in subdirs:
            process_single(sub, output_dir, args)
    else:
        process_single(traj_path, output_dir, args)


if __name__ == '__main__':
    main()
