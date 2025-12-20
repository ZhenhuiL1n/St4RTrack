import os
import glob
import numpy as np
import cv2
import torch
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from torchvision import transforms
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

class DNADataset(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location,
                 depth_path=None, # Optional override, otherwise assumes {frame}/depth_rendered
                 S=16, # Sequence length (number of views to load)
                 resolution=(512, 288),
                 transform=None,
                 *args, **kwargs):
        super().__init__(resolution=resolution, *args, **kwargs)
        self.dataset_location = dataset_location
        self.resolution = resolution
        self.S = S
        self.transform = transform if transform else transforms.ToTensor()
        
        # Discover frames
        # dataset_location should be .../0008_01
        self.frames = sorted([d for d in os.listdir(dataset_location) if d.isdigit() and os.path.isdir(os.path.join(dataset_location, d))], key=int)
        
        print(f"DNADataset: Found {len(self.frames)} frames in {dataset_location}")
        
        # Pre-scan views to ensure consistency?
        # We assume all frames have same views.
        if self.frames:
            sample_frame = os.path.join(dataset_location, self.frames[0])
            self.view_files = sorted(glob.glob(os.path.join(sample_frame, 'rgbs', '*.png')))
            self.num_available_views = len(self.view_files)
            print(f"DNADataset: Found {self.num_available_views} views per frame.")
            
    def __len__(self):
        return len(self.frames)

    def _get_views(self, index, resolution=None, rng=None):
        if resolution is None:
            resolution = self.resolution
            
        frame_name = self.frames[index]
        frame_dir = os.path.join(self.dataset_location, frame_name)
        
        rgb_dir = os.path.join(frame_dir, 'rgbs')
        depth_dir = os.path.join(frame_dir, 'depth_rendered')
        info_path = os.path.join(frame_dir, 'info.npz')
        
        # Load View Paths
        # We want to return S views.
        # If S < num_views, we should sample? Or return all? 
        # BaseStereoViewDataset usually handles random selection if we return list.
        # But here we construct the batch.
        # Let's return ALL views and let the collate/sampler handle it? 
        # No, typically _get_views returns specific B images.
        
        # For simplicity, let's pick S views (deterministically or random?). 
        # If we pick random here, we might break epoch consistency unless RNG is passed.
        # Let's verify standard DUSt3R behavior. CustomDUSt3R takes a sliding window.
        # DNA is a ring of cameras. 
        # Let's take a sliding window of S views.
        # Or just random S views.
        
        views_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        if len(views_files) > self.S:
            # Simple cyclic selection or random?
            # Let's take the first S for now to be safe, or stride.
            # Ideally we want the network to see all pairs.
            # But `_get_views` is called once per "Item".
            # If we want to train on different subsets, we should probably increase __len__ 
            # to be (NumFrames * NumSubsets).
            # For "Finetuning geometry", seeing all views is good.
            # Let's taking a random subset of S views using the RNG if provided.
            if rng is None:
                rng = np.random.RandomState(index)
            
            idxs = np.sort(rng.choice(len(views_files), self.S, replace=False))
            views_files = [views_files[i] for i in idxs]
        
        # Load Info
        info = np.load(info_path)
        intrinsics_all = info['intrinsics'] # [N, 3, 3]
        view_ids = info['view_ids']
        
        # Initialize Output Tensors
        B = len(views_files)
        W, H = resolution
        
        imgs = torch.zeros((B, 3, H, W), dtype=torch.float32)
        depthmaps = torch.zeros((B, H, W), dtype=torch.float32)
        valid_masks = torch.zeros((B, H, W), dtype=torch.bool)
        intrinsics_torch = torch.zeros((B, 3, 3), dtype=torch.float32)
        
        for i, rgb_path in enumerate(views_files):
            # RGB
            img = imread_cv2(rgb_path) # BGR
            # Resize
            img = cv2.resize(img, (W, H))
            imgs[i] = self.transform(img)
            
            # Identify View ID from filename '00xx.png'
            vid_str = os.path.splitext(os.path.basename(rgb_path))[0]
            try:
                vid = int(vid_str)
            except:
                vid = i # Fallback
                
            # Depth
            depth_path = os.path.join(depth_dir, f"{vid_str}.npy")
            if os.path.exists(depth_path):
                dmap = np.load(depth_path) # [H_orig, W_orig] or [H,W]?
                # Data generation saved matches original resolution? 
                # extract_dna used ratio 0.5. 
                # render_depth used point_cloud.ply (which frame?).
                # We need to ensure resolution match.
                dmap = cv2.resize(dmap, (W, H), interpolation=cv2.INTER_NEAREST)
                depthmaps[i] = torch.from_numpy(dmap)
                depthmaps[i] = torch.from_numpy(dmap)
                
            # Load Foreground Mask
            mask_path = os.path.join(frame_dir, 'masks', f"{vid_str}.png")
            if os.path.exists(mask_path):
                mask = imread_cv2(mask_path)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                # Assuming mask is 255 for foreground
                fg_mask = (mask > 127)
                if fg_mask.ndim == 3: fg_mask = fg_mask[..., 0] # Handle 3-channel masks
                valid_masks[i] = (depthmaps[i] > 0) & torch.from_numpy(fg_mask)
            else:
                valid_masks[i] = (depthmaps[i] > 0)
            
            # Intrinsics
            # Find index in info
            # info['view_ids'] matches vid
            try:
                # info['view_ids'] might be array of strings or ints
                # Based on previous code, they are ints
                v_arr_idx = np.where(view_ids == vid)[0][0]
                K = intrinsics_all[v_arr_idx].copy()
                
                # Adjust K for resize
                # Original K was for extracted image size.
                # If we resized to (W, H), we scale K.
                # How do we know original size? 
                # Info doesn't store H, W explicitly.
                # However, resize_image_depthmap_crop logic in DUSt3R is complex.
                # Here we forcibly resized to (W, H).
                # We assume the input image `img` (before resize) matched K.
                # Wait, `imread_cv2` returns original size.
                h_orig, w_orig = imread_cv2(rgb_path).shape[:2]
                
                scale_x = W / w_orig
                scale_y = H / h_orig
                K[0, :] *= scale_x
                K[1, :] *= scale_y
                
                intrinsics_torch[i] = torch.from_numpy(K)
                
            except Exception as e:
                print(f"Warning: Intrinsics error for view {vid}: {e}")
                
        # Prepare "views2" structure (Supervised)
        # views1 is usually reference or duplicate if we want standard stereo
        # CustomDUSt3R returns views1 (ref) and views2 (target).
        # But for MVS/DeMo, views2 is the main payload containing GT depth.
        
        # We dummy-fill views1 as it's often just used for "instance" labels or same as views2
        # But let's follow the CustomDUSt3R pattern where views1 is similar or meta-data.
        
        # Actually, for training, DUSt3R expects:
        # views usually is list of dicts or tuple of dicts (view1, view2).
        # BaseStereoViewDataset returns (view1, view2).
        
        # Let's clone views1 from views2 but maybe just first frame?
        # CustomDUSt3R: views1 has repeated Frame0. views2 has all frames (window).
        # Here we have Multiview. Frame0 is View0?
        # Let's just make views1 = views2 copy for now (symmetrical).
        
        views = {
            'img': imgs,
            'depthmap': depthmaps,
            'valid_mask': valid_masks,
            'camera_intrinsics': intrinsics_torch,
            'dataset': 'DNA',
            'label': [frame_name] * B,
            'instance': [f"{v}" for v in range(B)],
            'pts3d': torch.zeros((B, H, W, 3), dtype=torch.float32),
            'camera_pose': torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1),
            'traj_mask': torch.zeros((B, H, W), dtype=torch.bool),
            'traj_ptc': torch.zeros((B, H, W, 3), dtype=torch.float32),
            'supervised_label': torch.ones(B, dtype=torch.float32)
        }
        
        # DUMMY FIX: Set one pixel to True in traj_mask to avoid "Empty batch" error in training.py
        # Since traj_weight=0, this won't affect the loss.
        views['traj_mask'][:, 0, 0] = True
        
        return views, views


class DNAMultiSeqDataset(BaseStereoViewDataset):
    """
    DNA-Rendering Dataset for MULTIPLE sequences.
    
    Scans a parent directory containing multiple sequences (e.g., toy_dataset/Part1/)
    and creates training samples from all frames across all sequences.
    
    Loads BOTH:
    - GT rendered depth from depth_rendered/{view}.npy (foreground, multi-view consistent)
    - MoGe calibrated depth from calibrated_depth/{seq}/{seq}_view{view}/moge_calibrated.npy (full image)
    """
    
    def __init__(self,
                 dataset_location,  # Parent dir, e.g., toy_dataset/Part1
                 moge_depth_dir=None,  # Path to calibrated_depth/ folder (optional)
                 S=12,  # Number of views per sample
                 resolution=(512, 512),
                 stride=1,  # Frame stride within sequence
                 transform=None,
                 exclude_seqs=None,  # List of sequence names to exclude
                 view_sample_mode='random',  # 'random', 'sequential', 'opposite'
                 load_moge=True,  # Whether to load MoGe depth for background
                 *args, **kwargs):
        super().__init__(resolution=resolution, *args, **kwargs)
        self.dataset_location = dataset_location
        self.moge_depth_dir = moge_depth_dir
        self.resolution = resolution
        self.S = S
        self.stride = stride
        self.transform = transform if transform else transforms.ToTensor()
        self.view_sample_mode = view_sample_mode
        self.exclude_seqs = set(exclude_seqs or [])
        self.load_moge = load_moge
        
        # Discover all sequences
        self.sequences = []
        for seq_name in sorted(os.listdir(dataset_location)):
            seq_dir = os.path.join(dataset_location, seq_name)
            if not os.path.isdir(seq_dir):
                continue
            if seq_name in self.exclude_seqs:
                continue
            self.sequences.append(seq_name)
        
        print(f"DNAMultiSeqDataset: Found {len(self.sequences)} sequences")
        
        # Pre-load MoGe calibrated depth data
        self.moge_cache = {}
        if self.load_moge and self.moge_depth_dir:
            print(f"DNAMultiSeqDataset: Loading MoGe calibrated depth from {moge_depth_dir}")
            for seq_name in self.sequences:
                for view_idx in range(48):
                    moge_path = os.path.join(moge_depth_dir, seq_name, f"{seq_name}_view{view_idx}", "moge_calibrated.npy")
                    if os.path.exists(moge_path):
                        try:
                            self.moge_cache[(seq_name, view_idx)] = np.load(moge_path, allow_pickle=True).item()
                        except:
                            pass
            print(f"DNAMultiSeqDataset: Loaded {len(self.moge_cache)} MoGe calibrated files")
        
        # Build list of (seq_name, frame_idx) pairs
        self.samples = []
        for seq_name in self.sequences:
            seq_dir = os.path.join(dataset_location, seq_name)
            frames = [d for d in os.listdir(seq_dir) if d.isdigit() and 
                      os.path.isdir(os.path.join(seq_dir, d))]
            frames = sorted(frames, key=int)
            
            # Filter frames that have depth_rendered
            valid_frames = []
            for frame_name in frames:
                depth_dir = os.path.join(seq_dir, frame_name, 'depth_rendered')
                if os.path.exists(depth_dir) and len(os.listdir(depth_dir)) >= 48:
                    valid_frames.append(frame_name)
            
            # Add samples with stride
            for i in range(0, len(valid_frames), stride):
                self.samples.append((seq_name, valid_frames[i]))
        
        print(f"DNAMultiSeqDataset: Created {len(self.samples)} samples")
        
        # Check view count from first sample
        if self.samples:
            seq_name, frame_name = self.samples[0]
            rgb_dir = os.path.join(dataset_location, seq_name, frame_name, 'rgbs')
            self.num_available_views = len(glob.glob(os.path.join(rgb_dir, '*.png')))
            print(f"DNAMultiSeqDataset: {self.num_available_views} views per frame")
    
    def __len__(self):
        return len(self.samples)
    
    def _get_views(self, index, resolution=None, rng=None):
        if resolution is None:
            resolution = self.resolution
        
        seq_name, frame_name = self.samples[index]
        frame_dir = os.path.join(self.dataset_location, seq_name, frame_name)
        
        rgb_dir = os.path.join(frame_dir, 'rgbs')
        depth_dir = os.path.join(frame_dir, 'depth_rendered')
        cameras_path = os.path.join(frame_dir, 'cameras.json')
        
        # Get all views
        all_views = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        num_available = len(all_views)
        
        # Select S views based on mode
        if rng is None:
            rng = np.random.RandomState(index)
        
        if num_available <= self.S:
            selected_indices = list(range(num_available))
        elif self.view_sample_mode == 'random':
            selected_indices = np.sort(rng.choice(num_available, self.S, replace=False))
        elif self.view_sample_mode == 'sequential':
            start = rng.randint(0, max(1, num_available - self.S))
            selected_indices = list(range(start, start + self.S))
        elif self.view_sample_mode == 'opposite':
            # Sample views that are roughly opposite (e.g., 0 and 24 for 48 views)
            half = num_available // 2
            first_half = rng.choice(half, self.S // 2, replace=False)
            second_half = first_half + half
            selected_indices = np.sort(np.concatenate([first_half, second_half]))
        else:
            selected_indices = list(range(min(self.S, num_available)))
        
        views_files = [all_views[i] for i in selected_indices]
        
        # Load camera intrinsics
        import json
        intrinsics_dict = {}
        if os.path.exists(cameras_path):
            with open(cameras_path) as f:
                cameras = json.load(f)
            for cam in cameras:
                vid = int(cam['img_name'])
                K = np.array([
                    [cam['fx'], 0, cam['width'] / 2],
                    [0, cam['fy'], cam['height'] / 2],
                    [0, 0, 1]
                ])
                intrinsics_dict[vid] = K
        
        # Prepare output tensors
        B = len(views_files)
        W, H = resolution if isinstance(resolution, (list, tuple)) else (resolution, resolution)
        
        imgs = torch.zeros((B, 3, H, W), dtype=torch.float32)
        depthmaps = torch.zeros((B, H, W), dtype=torch.float32)  # GT rendered depth
        moge_depthmaps = torch.zeros((B, H, W), dtype=torch.float32)  # MoGe calibrated depth
        valid_masks = torch.zeros((B, H, W), dtype=torch.bool)  # Foreground mask
        bg_masks = torch.zeros((B, H, W), dtype=torch.bool)  # Background mask
        intrinsics_torch = torch.zeros((B, 3, 3), dtype=torch.float32)
        
        for i, rgb_path in enumerate(views_files):
            # Get view ID
            vid_str = os.path.splitext(os.path.basename(rgb_path))[0]
            vid = int(vid_str)
            
            # Load RGB
            img = imread_cv2(rgb_path)  # BGR
            h_orig, w_orig = img.shape[:2]
            img = cv2.resize(img, (W, H))
            imgs[i] = self.transform(img)
            
            # Load GT rendered depth (foreground only)
            depth_path = os.path.join(depth_dir, f"{vid_str}.npy")
            if os.path.exists(depth_path):
                dmap = np.load(depth_path)
                dmap = cv2.resize(dmap, (W, H), interpolation=cv2.INTER_NEAREST)
                depthmaps[i] = torch.from_numpy(dmap)
            
            # Load foreground mask
            mask_path = os.path.join(frame_dir, 'masks', f"{vid_str}.png")
            fg_mask_np = None
            if os.path.exists(mask_path):
                mask = imread_cv2(mask_path)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                fg_mask_np = (mask > 127)
                if fg_mask_np.ndim == 3:
                    fg_mask_np = fg_mask_np[..., 0]
                valid_masks[i] = (depthmaps[i] > 0) & torch.from_numpy(fg_mask_np)
                bg_masks[i] = torch.from_numpy(~fg_mask_np)
            else:
                valid_masks[i] = (depthmaps[i] > 0)
            
            # Load MoGe calibrated depth (full image)
            # Try cache first, then lazy load from disk
            if (seq_name, vid) in self.moge_cache:
                moge_data = self.moge_cache[(seq_name, vid)]
                frame_key = f"frame_{int(frame_name):04d}"
                if frame_key in moge_data:
                    moge_depth = moge_data[frame_key]['depth'].astype(np.float32)
                    if moge_depth.shape != (H, W):
                        moge_depth = cv2.resize(moge_depth, (W, H), interpolation=cv2.INTER_LINEAR)
                    # Handle NaN/Inf by median fill instead of zeroing
                    bad_mask = np.isnan(moge_depth) | np.isinf(moge_depth)
                    if bad_mask.any():
                        good_vals = moge_depth[~bad_mask]
                        if good_vals.size > 0:
                            fill_val = np.median(good_vals)
                            moge_depth = np.where(bad_mask, fill_val, moge_depth)
                        else:
                            moge_depth = np.zeros_like(moge_depth)
                    moge_depthmaps[i] = torch.from_numpy(moge_depth)
            elif self.moge_depth_dir is not None:
                # Lazy load from disk
                moge_path = os.path.join(self.moge_depth_dir, seq_name, f"{seq_name}_view{vid}", "moge_calibrated.npy")
                if os.path.exists(moge_path):
                    try:
                        moge_file = np.load(moge_path, allow_pickle=True).item()
                        frame_key = f"frame_{int(frame_name):04d}"
                        if frame_key in moge_file:
                            moge_depth = moge_file[frame_key]['depth'].astype(np.float32)
                            if moge_depth.shape != (H, W):
                                moge_depth = cv2.resize(moge_depth, (W, H), interpolation=cv2.INTER_LINEAR)
                            
                            # Handle NaN/Inf by inpainting instead of zeroing
                            bad_mask = np.isnan(moge_depth) | np.isinf(moge_depth)
                            if bad_mask.any():
                                # Replace bad values with median temporarily
                                good_vals = moge_depth[~bad_mask]
                                if good_vals.size > 0:
                                    fill_val = np.median(good_vals)
                                    moge_depth = np.where(bad_mask, fill_val, moge_depth)
                                else:
                                    moge_depth = np.zeros_like(moge_depth)  # Fallback if all bad
                            
                            moge_depthmaps[i] = torch.from_numpy(moge_depth)
                    except Exception as e:
                        pass  # Skip if load fails
            
            # Intrinsics
            if vid in intrinsics_dict:
                K = intrinsics_dict[vid].copy()
                scale_x = W / w_orig
                scale_y = H / h_orig
                K[0, :] *= scale_x
                K[1, :] *= scale_y
                intrinsics_torch[i] = torch.from_numpy(K)
        
        # Compute pts3d from MoGe depth (like DNASingleSeqDataset)
        pts3d_moge = torch.zeros((B, H, W, 3), dtype=torch.float32)
        for i in range(B):
            # Use moge_depthmaps if available, otherwise depthmaps (GT)
            depth_to_use = moge_depthmaps[i] if moge_depthmaps[i].any() else depthmaps[i]
            mask_to_use = valid_masks[i] if depth_to_use is depthmaps[i] else (moge_depthmaps[i] > 0)
            if mask_to_use.any() and intrinsics_torch[i].any():
                pts, _ = depthmap_to_absolute_camera_coordinates(
                    depth_to_use.numpy(), 
                    intrinsics_torch[i].numpy(), 
                    camera_pose=None, 
                    proj_mode='depth'
                )
                pts3d_moge[i] = torch.from_numpy(pts)
        
        # views1: Reference view (minimal fields, repeated first frame)
        # Following CustomDUSt3R pattern
        views1 = {
            'img': imgs[0:1].repeat(B, 1, 1, 1),  # First frame repeated
            'camera_intrinsics': intrinsics_torch[0:1].repeat(B, 1, 1),
            'dataset': 'DNA',
            'label': [f"{seq_name}/{frame_name}"] * B,
            'instance': [os.path.splitext(os.path.basename(views_files[0]))[0]] * B,
            'supervised_label': torch.ones(B, dtype=torch.float32),
            # Required for loss computation
            'traj_mask': valid_masks.clone(),
            'traj_ptc': torch.zeros((B, H, W, 3), dtype=torch.float32),
            'pts3d': pts3d_moge[0:1].repeat(B, 1, 1, 1),  # Use first view's pts3d
            'valid_mask': valid_masks.clone(),
            'camera_pose': torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1),
        }
        
        # views2: Target views with depth and all data
        views2 = {
            'img': imgs,
            'img_org': imgs.clone(),  # Required by CustomDUSt3R format
            'depthmap': depthmaps,  # GT rendered depth (foreground)
            'moge_depth': moge_depthmaps,  # MoGe calibrated depth (full)
            'valid_mask': valid_masks,  # Mask for GT depth (foreground only)
            'bg_mask': bg_masks,  # Background mask for weak supervision
            'camera_intrinsics': intrinsics_torch,
            'dataset': 'DNA',
            'label': [f"{seq_name}/{frame_name}"] * B,
            'instance': [f"{vid_str}" for vid_str in [os.path.splitext(os.path.basename(f))[0] for f in views_files]],
            'pts3d_moge': pts3d_moge,  # 3D points computed from depth
            'pts3d': pts3d_moge.clone(),  # Also set pts3d for compatibility
            'camera_pose': torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1),
            'supervised_label': torch.ones(B, dtype=torch.float32),
        }
        
        return views1, views2


class DNASingleSeqDataset(BaseStereoViewDataset):
    """
    DNA-Rendering Dataset for a SINGLE sequence with all 48 views.
    For TTA (Test-Time Adaptation) on a specific sequence.
    
    Loads calibrated MoGe depth from calibrated_depth/{seq}/{seq}_view{v}/moge_calibrated.npy
    """
    
    def __init__(self,
                 dataset_location,  # Sequence dir, e.g., toy_dataset/Part1/0012_09
                 moge_depth_dir=None,  # Path to calibrated_depth/ folder
                 S=8,  # Number of views per sample
                 resolution=(512, 512),
                 transform=None,
                 view_sample_mode='random',  # 'random', 'sequential', 'opposite'
                 load_moge=True,  # Whether to load MoGe calibrated depth
                 *args, **kwargs):
        super().__init__(resolution=resolution, *args, **kwargs)
        self.dataset_location = dataset_location
        self.moge_depth_dir = moge_depth_dir
        self.resolution = resolution
        self.S = S
        self.transform = transform if transform else transforms.ToTensor()
        self.view_sample_mode = view_sample_mode
        self.load_moge = load_moge
        
        # Get sequence name from path
        self.seq_name = os.path.basename(dataset_location.rstrip('/'))
        
        # Discover frames
        self.frames = sorted([d for d in os.listdir(dataset_location) if d.isdigit() and 
                      os.path.isdir(os.path.join(dataset_location, d))], key=int)
        
        print(f"DNASingleSeqDataset: Found {len(self.frames)} frames in {self.seq_name}")
        
        # Get number of views from first frame
        if self.frames:
            rgb_dir = os.path.join(dataset_location, self.frames[0], 'rgbs')
            self.num_available_views = len(glob.glob(os.path.join(rgb_dir, '*.png')))
            print(f"DNASingleSeqDataset: {self.num_available_views} views per frame")
        
        # Pre-load MoGe calibrated depth data
        self.moge_cache = {}
        if self.load_moge and self.moge_depth_dir:
            print(f"DNASingleSeqDataset: Loading MoGe calibrated depth from {moge_depth_dir}")
            for view_idx in range(48):
                moge_path = os.path.join(moge_depth_dir, self.seq_name, f"{self.seq_name}_view{view_idx}", "moge_calibrated.npy")
                if os.path.exists(moge_path):
                    try:
                        self.moge_cache[view_idx] = np.load(moge_path, allow_pickle=True).item()
                    except Exception as e:
                        print(f"Error loading {moge_path}: {e}")
            print(f"DNASingleSeqDataset: Loaded {len(self.moge_cache)} MoGe calibrated files")
    
    def __len__(self):
        return len(self.frames)
    
    def _get_views(self, index, resolution=None, rng=None):
        if resolution is None:
            resolution = self.resolution
        
        frame_name = self.frames[index]
        frame_dir = os.path.join(self.dataset_location, frame_name)
        
        rgb_dir = os.path.join(frame_dir, 'rgbs')
        cameras_path = os.path.join(frame_dir, 'cameras.json')
        
        # Get all views
        all_views = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        num_available = len(all_views)
        
        # Select S views based on mode
        if rng is None:
            rng = np.random.RandomState(index)
        
        if num_available <= self.S:
            selected_indices = list(range(num_available))
        elif self.view_sample_mode == 'random':
            selected_indices = np.sort(rng.choice(num_available, self.S, replace=False))
        elif self.view_sample_mode == 'sequential':
            start = rng.randint(0, max(1, num_available - self.S))
            selected_indices = list(range(start, start + self.S))
        elif self.view_sample_mode == 'opposite':
            half = num_available // 2
            first_half = rng.choice(half, self.S // 2, replace=False)
            second_half = first_half + half
            selected_indices = np.sort(np.concatenate([first_half, second_half]))
        else:
            selected_indices = list(range(min(self.S, num_available)))
        
        views_files = [all_views[i] for i in selected_indices]
        
        # Load camera intrinsics
        import json
        intrinsics_dict = {}
        if os.path.exists(cameras_path):
            with open(cameras_path) as f:
                cameras = json.load(f)
            for cam in cameras:
                vid = int(cam['img_name'])
                K = np.array([
                    [cam['fx'], 0, cam['width'] / 2],
                    [0, cam['fy'], cam['height'] / 2],
                    [0, 0, 1]
                ])
                intrinsics_dict[vid] = K
        
        # Prepare output tensors
        B = len(views_files)
        W, H = resolution if isinstance(resolution, (list, tuple)) else (resolution, resolution)
        
        imgs = torch.zeros((B, 3, H, W), dtype=torch.float32)
        depthmaps = torch.zeros((B, H, W), dtype=torch.float32)  # Calibrated MoGe depth
        valid_masks = torch.zeros((B, H, W), dtype=torch.bool)
        intrinsics_torch = torch.zeros((B, 3, 3), dtype=torch.float32)
        
        for i, rgb_path in enumerate(views_files):
            vid_str = os.path.splitext(os.path.basename(rgb_path))[0]
            vid = int(vid_str)
            
            # Load RGB
            img = imread_cv2(rgb_path)
            h_orig, w_orig = img.shape[:2]
            img = cv2.resize(img, (W, H))
            imgs[i] = self.transform(img)
            
            # Load MoGe calibrated depth
            if self.load_moge and vid in self.moge_cache:
                moge_data = self.moge_cache[vid]
                frame_key = f"frame_{int(frame_name):04d}"
                if frame_key in moge_data:
                    moge_depth = moge_data[frame_key]['depth'].copy()
                    moge_mask = moge_data[frame_key].get('mask', np.ones_like(moge_depth, dtype=bool))
                    
                    # Handle NaN values - replace with 0 and mark as invalid
                    nan_mask = np.isnan(moge_depth) | np.isinf(moge_depth)
                    moge_depth = np.nan_to_num(moge_depth, nan=0.0, posinf=0.0, neginf=0.0)
                    moge_mask = moge_mask & (~nan_mask)
                    
                    if moge_depth.shape != (H, W):
                        moge_depth = cv2.resize(moge_depth, (W, H), interpolation=cv2.INTER_LINEAR)
                        moge_mask = cv2.resize(moge_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    depthmaps[i] = torch.from_numpy(moge_depth.astype(np.float32))
                    valid_masks[i] = torch.from_numpy(moge_mask) & (depthmaps[i] > 0)
            
            # Intrinsics
            if vid in intrinsics_dict:
                K = intrinsics_dict[vid].copy()
                scale_x = W / w_orig
                scale_y = H / h_orig
                K[0, :] *= scale_x
                K[1, :] *= scale_y
                intrinsics_torch[i] = torch.from_numpy(K)
        
        # Compute pts3d_moge from depth (like CustomDUSt3R) - BEFORE views1
        pts3d_moge = torch.zeros((B, H, W, 3), dtype=torch.float32)
        for i in range(B):
            if valid_masks[i].any():
                pts, _ = depthmap_to_absolute_camera_coordinates(
                    depthmaps[i].numpy(), 
                    intrinsics_torch[i].numpy(), 
                    camera_pose=None, 
                    proj_mode='depth'
                )
                pts3d_moge[i] = torch.from_numpy(pts)
        
        # views1: Reference view (minimal fields, repeated first frame)
        views1 = {
            'img': imgs[0:1].repeat(B, 1, 1, 1),
            'camera_intrinsics': intrinsics_torch[0:1].repeat(B, 1, 1),
            'dataset': 'DNA',
            'label': [f"{self.seq_name}/{frame_name}"] * B,
            'instance': [os.path.splitext(os.path.basename(views_files[0]))[0]] * B,
            'supervised_label': torch.ones(B, dtype=torch.float32),
            'traj_mask': valid_masks.clone(),
            'traj_ptc': torch.zeros((B, H, W, 3), dtype=torch.float32),
            'pts3d': pts3d_moge[0:1].repeat(B, 1, 1, 1),  # Use first view's pts3d
            'valid_mask': valid_masks.clone(),
            'camera_pose': torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1),
        }
        
        # views2: Target views with depth and all data
        views2 = {
            'img': imgs,
            'img_org': imgs.clone(),
            'depthmap': depthmaps,  # Calibrated MoGe depth
            'valid_mask': valid_masks,
            'camera_intrinsics': intrinsics_torch,
            'dataset': 'DNA',
            'label': [f"{self.seq_name}/{frame_name}"] * B,
            'instance': [f"{vid_str}" for vid_str in [os.path.splitext(os.path.basename(f))[0] for f in views_files]],
            'pts3d_moge': pts3d_moge,  # 3D points computed from depth
            'pts3d': pts3d_moge.clone(),  # Also set pts3d for compatibility
            'camera_pose': torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1),
            'supervised_label': torch.ones(B, dtype=torch.float32),
        }
        
        return views1, views2
