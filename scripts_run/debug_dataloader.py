
import sys
import os
sys.path.append(os.getcwd()) # Ensure root is in path

import torch
from dust3r.datasets.dna import DNADataset
from torchvision import transforms

def debug_dataset():
    # Arguments matching train_geometry.sh
    data_root = "/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/Part1/0008_01"
    
    print("Initializing DNADataset...")
    dataset = DNADataset(
        dataset_location=data_root,
        S=12,
        resolution=(512, 288),
        transform=transforms.ToTensor()
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Simulate DataLoader loop with single worker
    # We can iterate manually
    for idx in range(len(dataset)):
        print(f"\n--- Loading Index {idx} ---")
        try:
            # Emulate BaseStereoViewDataset __getitem__ logic if needed, 
            # but DNADataset inherits from it.
            # BaseStereoViewDataset.__getitem__ calls _get_views.
            
            # Use the public interface
            views = dataset[idx] 
            
            # Verify structure
            print(f"Successfully loaded views (len={len(views)})")
            
            # views should be a list of dicts (if BaseStereoViewDataset logic holds)
            # OR a tuple of dicts (if my previous analysis of 'views, views' return was kept)
            
            for i, view in enumerate(views):
                print(f"View {i} keys: {view.keys()}")
                if 'img' in view:
                    print(f"  img shape: {view['img'].shape}")
                if 'depthmap' in view:
                    print(f"  depthmap shape: {view['depthmap'].shape}")
                if 'valid_mask' in view:
                    print(f"  valid_mask shape: {view['valid_mask'].shape}")
                    
            # Check for Assertion failure conditions
            # assert len(views) == dataset.num_views
            print(f"dataset.num_views: {dataset.num_views}")
            assert len(views) == dataset.num_views
            
            break # Just check first item
            
        except Exception as e:
            print(f"Error loading index {idx}:")
            print(e)
            import traceback
            traceback.print_exc()
            print("\nDropping into pdb...")
            import pdb; pdb.set_trace()
            break

if __name__ == "__main__":
    debug_dataset()
