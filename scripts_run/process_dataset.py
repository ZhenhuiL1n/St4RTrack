import os
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import sys
import glob

def find_ply_path_robust(frame_path):
    # Try 30000 first, then 20000, or any
    # frame_path is in PROCESSED structure
    base = frame_path / "point_cloud"
    if not base.exists():
        return None
    
    # Check specific iterations
    for it in [30000, 20000, 7000]:
        check = base / f"iteration_{it}" / "point_cloud.ply"
        if check.exists():
            return check
            
    # Fallback to any iteration
    others = list(base.glob("iteration_*/point_cloud.ply"))
    if others:
        return others[0] # Pick first found
        
    return None

def process_dataset(extracted_root, processed_root):
    extracted_root = Path(extracted_root)
    processed_root = Path(processed_root)
    
    if not extracted_root.exists():
        print(f"Error: Extracted root {extracted_root} does not exist.")
        return
    if not processed_root.exists():
        print(f"Error: Processed root {processed_root} does not exist.")
        return

    # Find scenes in EXTRACTED root (these are the ones we want to process)
    # Filter out _annots
    scenes = [d for d in extracted_root.iterdir() if d.is_dir() and "_annots" not in d.name and "annotations" not in d.name]
    scenes = sorted(scenes)
    
    print(f"Found {len(scenes)} scenes in {extracted_root}")
    
    tasks = []
    
    for scene in scenes:
        seq_name = scene.name
        scene_processed = processed_root / seq_name
        
        if not scene_processed.exists():
            print(f"Warning: Corresponding processed scene {seq_name} not found in {processed_root}")
            continue
            
        # Find all frames (numbered directories)
        frames = [d for d in scene.iterdir() if d.is_dir() and d.name.isdigit()]
        frames = sorted(frames, key=lambda x: int(x.name))
        
        for frame in frames:
            frame_name = frame.name
            frame_processed = scene_processed / frame_name
            
            # Find PLY in PROCESSED
            ply_path = find_ply_path_robust(frame_processed)
            
            # Find cameras.json in PROCESSED
            cameras_json = frame_processed / "cameras.json"
            
            output_dir = frame / "depth_rendered"
            
            if ply_path and cameras_json.exists():
                 # Check if need to render (skip if done?)
                 # Simple check: if dir exists and has 48 files? 
                 # Let's just blindly add to tasks or check 1 file.
                 # User asked to render "rest", assuming some done.
                 # But re-rendering is safe.
                 
                 tasks.append({
                    "ply_path": str(ply_path),
                    "cameras_json": str(cameras_json),
                    "output_dir": str(output_dir),
                    "scene": seq_name,
                    "frame": frame_name
                })
            else:
                # pass
                # print(f"Missing PLY/Cam for {seq_name}/{frame_name}")
                pass
    
    print(f"Found {len(tasks)} frames to process.")
    
    from multiprocessing import Pool, cpu_count
    
    # Use 80% of available cores
    num_workers = max(1, int(cpu_count() * 0.8))
    # Cap workers because rendering uses GPU or CPU intesively? 
    # "render_depth.py" doing "Manual Splatting" on CPU (numpy/cv2).
    # 20 workers might be OK.
    
    print(f"Using {num_workers} workers for parallel processing.")
    
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_frame, tasks), total=len(tasks)))

def process_frame(task):
    # Construct command
    # Warning: render_depth.py usually expects to run from root or relative imports might break
    # We should run it as module or from root logic.
    # calling "python scripts_run/render_depth.py" assumes cwd is root.
    
    cmd = [
        "python", "scripts_run/render_depth.py",
        "--ply_path", task["ply_path"],
        "--cameras_json", task["cameras_json"],
        "--output_dir", task["output_dir"],
        "--point_size", "2.0" # Changed from 3.0 to 2.0 per previous analysis/preference for finer detail?
    ]
    
    # Run command
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # print(f"Error processing {task['scene']}/{task['frame']}: {e.stderr.decode()}")
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_root', required=True, help='Path to extracted root (e.g. .../toy_dataset/Part1)')
    parser.add_argument('--processed_root', required=True, help='Path to processed root (e.g. .../DNA_Processed/Part1)')
    args = parser.parse_args()
    
    process_dataset(args.extracted_root, args.processed_root)
