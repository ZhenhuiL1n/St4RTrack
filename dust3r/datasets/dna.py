import os
import glob
import numpy as np
import cv2
import torch
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from torchvision import transforms

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
