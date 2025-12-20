"""
DNA Multi-View Batch Dataset for TTA.

Ensures each batch of size N contains N different views from the SAME timestamp.
Works with CustomDUSt3R format.
"""

import os
import glob
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, Sampler
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from torchvision import transforms
import dust3r.datasets.utils.cropping as cropping


class DNAMultiViewBatchDataset(BaseStereoViewDataset):
    """
    DNA dataset where each sample is ONE view from ONE timestamp.
    When used with batch_size=8, you get 8 views from the same timestamp.
    
    Use with DNAMultiViewSampler to ensure batch grouping by timestamp.
    """
    
    def __init__(self,
                 dataset_location,  # Sequence dir, e.g., Part1/0012_09
                 moge_depth_dir,    # calibrated_depth folder
                 resolution=(512, 512),
                 transform=None,
                 *args, **kwargs):
        super().__init__(resolution=resolution, *args, **kwargs)
        self.dataset_location = dataset_location
        self.moge_depth_dir = moge_depth_dir
        self.resolution = resolution
        self.transform = transform if transform else transforms.ToTensor()
        
        self.seq_name = os.path.basename(dataset_location.rstrip('/'))
        
        # Discover timestamps
        self.timestamps = sorted([d for d in os.listdir(dataset_location) 
                                  if d.isdigit() and os.path.isdir(os.path.join(dataset_location, d))], key=int)
        
        print(f"DNAMultiViewBatchDataset: {len(self.timestamps)} timestamps in {self.seq_name}")
        
        # Pre-load MoGe depth
        print("Loading MoGe depth...")
        self.moge_cache = {}
        for view_idx in range(48):
            moge_path = os.path.join(moge_depth_dir, self.seq_name, f"{self.seq_name}_view{view_idx}", "moge_calibrated.npy")
            if os.path.exists(moge_path):
                try:
                    self.moge_cache[view_idx] = np.load(moge_path, allow_pickle=True).item()
                except:
                    pass
        print(f"Loaded {len(self.moge_cache)} view depth files")
        
        # Load intrinsics
        cameras_path = os.path.join(dataset_location, self.timestamps[0], "cameras.json")
        self.intrinsics_dict = {}
        if os.path.exists(cameras_path):
            with open(cameras_path) as f:
                cameras = json.load(f)
            for cam in cameras:
                vid = int(cam['img_name'])
                self.intrinsics_dict[vid] = np.array([
                    [cam['fx'], 0, cam['width'] / 2],
                    [0, cam['fy'], cam['height'] / 2],
                    [0, 0, 1]
                ], dtype=np.float32)
        
        # Build sample list: (timestamp_idx, view_idx)
        self.samples = []
        for t_idx, timestamp in enumerate(self.timestamps):
            for v_idx in range(48):
                self.samples.append((t_idx, v_idx))
        
        print(f"DNAMultiViewBatchDataset: {len(self.samples)} total samples (timestamps × views)")
    
    def __len__(self):
        return len(self.samples)
    
    def get_timestamp_count(self):
        return len(self.timestamps)
    
    def _get_views(self, index, resolution=None, rng=None):
        if resolution is None:
            resolution = self.resolution
        
        t_idx, view_idx = self.samples[index]
        timestamp = self.timestamps[t_idx]
        
        W, H = resolution if isinstance(resolution, (list, tuple)) else (resolution, resolution)
        
        # Load image
        img_path = os.path.join(self.dataset_location, timestamp, "rgbs", f"{view_idx:04d}.png")
        img = imread_cv2(img_path)
        h_orig, w_orig = img.shape[:2]
        img = cv2.resize(img, (W, H))
        img_tensor = self.transform(img)
        
        # Load depth
        depth = np.zeros((H, W), dtype=np.float32)
        mask = np.zeros((H, W), dtype=np.float32)
        
        frame_key = f"frame_{int(timestamp):04d}"
        if view_idx in self.moge_cache and frame_key in self.moge_cache[view_idx]:
            entry = self.moge_cache[view_idx][frame_key]
            depth = entry['depth'].copy().astype(np.float32)
            mask = entry.get('mask', np.ones_like(depth, dtype=np.float32))
            if mask.dtype == bool:
                mask = mask.astype(np.float32)
            
            # Handle NaN/Inf
            bad = np.isnan(depth) | np.isinf(depth)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            mask = mask * (~bad).astype(np.float32)
            
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Intrinsics
        K = self.intrinsics_dict.get(view_idx, np.eye(3, dtype=np.float32)).copy()
        scale_x, scale_y = W / w_orig, H / h_orig
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        
        # Build view dicts (B=1 for single view)
        views1 = {
            'img': img_tensor.unsqueeze(0),
            'camera_intrinsics': torch.from_numpy(K).unsqueeze(0),
            'dataset': 'DNA',
            'label': [f"{self.seq_name}/{timestamp}"],
            'instance': [f"{view_idx:04d}"],
            'supervised_label': torch.ones(1, dtype=torch.float32),
            'traj_mask': torch.from_numpy(mask > 0).unsqueeze(0),
            'traj_ptc': torch.zeros((1, H, W, 3), dtype=torch.float32),
            'pts3d': torch.zeros((1, H, W, 3), dtype=torch.float32),
            'valid_mask': torch.from_numpy(mask > 0).unsqueeze(0),
            'camera_pose': torch.eye(4, dtype=torch.float32).unsqueeze(0),
        }
        
        views2 = {
            'img': img_tensor.unsqueeze(0),
            'img_org': img_tensor.unsqueeze(0),
            'depthmap': torch.from_numpy(depth).unsqueeze(0),
            'valid_mask': torch.from_numpy(mask > 0).unsqueeze(0),
            'camera_intrinsics': torch.from_numpy(K).unsqueeze(0),
            'dataset': 'DNA',
            'label': [f"{self.seq_name}/{timestamp}"],
            'instance': [f"{view_idx:04d}"],
            'pts3d': torch.zeros((1, H, W, 3), dtype=torch.float32),
            'camera_pose': torch.eye(4, dtype=torch.float32).unsqueeze(0),
            'supervised_label': torch.ones(1, dtype=torch.float32),
        }
        
        return views1, views2


class DNAMultiViewSampler(Sampler):
    """
    Sampler that groups samples by timestamp.
    Each batch contains `batch_size` different views from the SAME timestamp.
    """
    
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.num_timestamps = dataset.get_timestamp_count()
        self.views_per_timestamp = 48
        
    def __iter__(self):
        # For each timestamp, sample batch_size views
        timestamp_order = list(range(self.num_timestamps))
        if self.shuffle:
            np.random.shuffle(timestamp_order)
        
        for t_idx in timestamp_order:
            # Sample batch_size views from this timestamp
            view_indices = np.random.choice(48, self.batch_size, replace=False)
            
            # Convert to global sample indices
            for v_idx in view_indices:
                yield t_idx * 48 + v_idx
    
    def __len__(self):
        return self.num_timestamps * self.batch_size
