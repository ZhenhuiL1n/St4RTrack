#!/usr/bin/env python3
"""
Validate DNA dataset for training - checks for bad data before training.

Checks:
1. Missing depth files
2. NaN/Inf values in depth
3. All-zero depth maps
4. MoGe calibration files consistency
"""

import os
import sys
import numpy as np
import glob
from tqdm import tqdm
import argparse

def check_moge_file(moge_path, seq_name, view_idx, verbose=False):
    """Check a single MoGe calibrated depth file for problems."""
    issues = []
    
    try:
        data = np.load(moge_path, allow_pickle=True).item()
    except Exception as e:
        return [f"Failed to load: {e}"]
    
    num_frames = len([k for k in data.keys() if k.startswith('frame_')])
    
    for frame_key in data.keys():
        if not frame_key.startswith('frame_'):
            continue
            
        frame_data = data[frame_key]
        
        if 'depth' not in frame_data:
            issues.append(f"{frame_key}: missing 'depth' key")
            continue
            
        depth = frame_data['depth']
        
        # Check for NaN
        nan_count = np.isnan(depth).sum()
        if nan_count > 0:
            nan_pct = nan_count / depth.size * 100
            if nan_pct > 50:
                issues.append(f"{frame_key}: {nan_pct:.1f}% NaN values")
        
        # Check for Inf
        inf_count = np.isinf(depth).sum()
        if inf_count > 0:
            inf_pct = inf_count / depth.size * 100
            if inf_pct > 50:
                issues.append(f"{frame_key}: {inf_pct:.1f}% Inf values")
        
        # Check for all zeros (after removing NaN/Inf)
        valid = ~(np.isnan(depth) | np.isinf(depth))
        if valid.sum() > 0:
            valid_depth = depth[valid]
            if np.allclose(valid_depth, 0):
                issues.append(f"{frame_key}: all valid values are zero")
        else:
            issues.append(f"{frame_key}: no valid depth values")
        
        # Check for negative depth
        neg_count = (depth < 0).sum()
        if neg_count > 0:
            neg_pct = neg_count / depth.size * 100
            if neg_pct > 10:
                issues.append(f"{frame_key}: {neg_pct:.1f}% negative depth")
    
    return issues


def validate_sequence(seq_name, data_dir, moge_dir):
    """Validate a single sequence."""
    seq_issues = {
        'missing_rgb': [],
        'missing_depth_rendered': [],
        'missing_moge': [],
        'moge_issues': {},
        'total_frames': 0,
        'total_views': 0,
    }
    
    seq_path = os.path.join(data_dir, seq_name)
    if not os.path.isdir(seq_path):
        return None
    
    # Get frames
    frames = [d for d in os.listdir(seq_path) if d.isdigit() and os.path.isdir(os.path.join(seq_path, d))]
    frames = sorted(frames, key=int)
    seq_issues['total_frames'] = len(frames)
    
    # Check each view's MoGe file
    for view_idx in range(48):
        moge_path = os.path.join(moge_dir, seq_name, f"{seq_name}_view{view_idx}", "moge_calibrated.npy")
        
        if not os.path.exists(moge_path):
            seq_issues['missing_moge'].append(view_idx)
            continue
        
        issues = check_moge_file(moge_path, seq_name, view_idx)
        if issues:
            seq_issues['moge_issues'][view_idx] = issues
    
    seq_issues['total_views'] = 48 - len(seq_issues['missing_moge'])
    
    # Check depth_rendered
    if frames:
        sample_frame = frames[0]
        depth_dir = os.path.join(seq_path, sample_frame, 'depth_rendered')
        if not os.path.exists(depth_dir):
            seq_issues['missing_depth_rendered'].append(sample_frame)
        else:
            depth_files = glob.glob(os.path.join(depth_dir, '*.npy'))
            if len(depth_files) < 48:
                seq_issues['missing_depth_rendered'].append(f"{sample_frame} (only {len(depth_files)} views)")
    
    return seq_issues


def main():
    parser = argparse.ArgumentParser(description="Validate DNA dataset for training")
    parser.add_argument("--data_dir", default="/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/Part1",
                        help="Path to DNA Part1 folder")
    parser.add_argument("--moge_dir", default="/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/calibrated_depth",
                        help="Path to calibrated_depth folder")
    parser.add_argument("--verbose", action="store_true", help="Show detailed issues")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DNA Dataset Validation")
    print("=" * 60)
    print(f"Data dir: {args.data_dir}")
    print(f"MoGe dir: {args.moge_dir}")
    print()
    
    # Get all sequences
    sequences = sorted([d for d in os.listdir(args.data_dir) 
                       if os.path.isdir(os.path.join(args.data_dir, d))])
    
    print(f"Found {len(sequences)} sequences")
    print()
    
    good_seqs = []
    bad_seqs = []
    
    for seq_name in tqdm(sequences, desc="Validating"):
        issues = validate_sequence(seq_name, args.data_dir, args.moge_dir)
        
        if issues is None:
            continue
        
        has_problems = (
            len(issues['missing_moge']) > 0 or
            len(issues['missing_depth_rendered']) > 0 or
            len(issues['moge_issues']) > 0
        )
        
        if has_problems:
            bad_seqs.append((seq_name, issues))
        else:
            good_seqs.append(seq_name)
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n✓ GOOD sequences ({len(good_seqs)}):")
    for seq in good_seqs:
        print(f"  {seq}")
    
    print(f"\n✗ BAD sequences ({len(bad_seqs)}):")
    for seq_name, issues in bad_seqs:
        print(f"\n  {seq_name}:")
        if issues['missing_moge']:
            print(f"    - Missing MoGe for views: {issues['missing_moge'][:5]}{'...' if len(issues['missing_moge']) > 5 else ''}")
        if issues['missing_depth_rendered']:
            print(f"    - Missing depth_rendered: {issues['missing_depth_rendered']}")
        if issues['moge_issues']:
            problem_views = list(issues['moge_issues'].keys())[:5]
            print(f"    - MoGe issues in views: {problem_views}{'...' if len(issues['moge_issues']) > 5 else ''}")
            if args.verbose:
                for view_idx, view_issues in list(issues['moge_issues'].items())[:3]:
                    for issue in view_issues[:2]:
                        print(f"        view{view_idx}: {issue}")
    
    print()
    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    if bad_seqs:
        bad_names = [s[0] for s in bad_seqs]
        print(f"\nExclude these sequences from training:")
        print(f"  exclude_seqs={bad_names}")
    else:
        print("\nAll sequences look good!")
    
    print()


if __name__ == "__main__":
    main()
