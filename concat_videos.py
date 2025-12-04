import os
import subprocess
import glob

# Define directories
DNA_SEQ_DIR = '/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/eval_benchmark/DNA_Seq'
LAB_SEQ_DIR = '/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/eval_benchmark/lab_Seq'
OUTPUT_DIR = '/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/all_videos_rotated'
RESULT_DIR = '/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/concatenated_results'

# Create result directory if it doesn't exist
os.makedirs(RESULT_DIR, exist_ok=True)

def get_duration(file_path):
    """Get the duration of a video file using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', file_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return None

def find_source_video(output_filename):
    """Find the corresponding source video for a given output filename."""
    if 'DNA_Seq' in output_filename:
        # Example: ...DNA_SeqDNA_01.mp4_0_1_48.mp4
        parts = output_filename.split('DNA_Seq')
        if len(parts) > 1:
            suffix = parts[1]
            # Try to match with files in DNA_SEQ_DIR
            for f in os.listdir(DNA_SEQ_DIR):
                if suffix.startswith(f):
                    return os.path.join(DNA_SEQ_DIR, f)
    elif 'lab_Seq' in output_filename:
        # Example: ...lab_Seqcalibrated_ari.mp4_0_1_48.mp4
        parts = output_filename.split('lab_Seq')
        if len(parts) > 1:
            suffix = parts[1]
            for f in os.listdir(LAB_SEQ_DIR):
                if suffix.startswith(f):
                    return os.path.join(LAB_SEQ_DIR, f)
    return None

def process_video(output_video_path):
    filename = os.path.basename(output_video_path)
    source_video_path = find_source_video(filename)
    
    if not source_video_path:
        print(f"Could not find source video for {filename}")
        return

    print(f"Processing: {filename}")
    print(f"  Source: {source_video_path}")
    
    # Get duration of the output video (the one in all_videos_rotated)
    duration = get_duration(output_video_path)
    if duration is None:
        print("  Failed to get duration.")
        return

    output_result_path = os.path.join(RESULT_DIR, filename)
    
    # Construct ffmpeg command
    # 1. Input 0: Source video (to be trimmed)
    # 2. Input 1: Output video (to be cropped)
    # Filter:
    # [1:v]crop=min(iw\,ih):min(iw\,ih):(iw-ow)/2:(ih-oh)/2[right]  -> Center crop output to square
    # [0:v]trim=duration={duration},setpts=PTS-STARTPTS[left]       -> Trim input to duration
    # [left][right]scale2ref=h=ih:w=iw*main_w/main_h[left_scaled][right_ref];[left_scaled][right_ref]hstack
    # Note: scale2ref might be complex. Let's try a simpler approach first:
    # Scale input to match output height? 
    # Let's assume we want to keep the output's cropped height.
    
    # Simplified filter:
    # [1:v]crop=min(iw\,ih):min(iw\,ih):(iw-ow)/2:(ih-oh)/2[right];
    # [0:v]trim=duration={duration},setpts=PTS-STARTPTS,scale=-1:'min(iw,ih)'[left];
    # [left][right]hstack
    
    # Wait, we can't access 'min(iw,ih)' of the *other* stream easily in a single filter chain without scale2ref or complex logical.
    # But we can just use scale2ref.
    
    filter_complex = (
        f"[1:v]crop=min(iw\\,ih):min(iw\\,ih):(iw-ow)/2:(ih-oh)/2[right];"
        f"[0:v]trim=duration={duration},setpts=PTS-STARTPTS[left_trimmed];"
        f"[left_trimmed][right]scale2ref=h=ih:w=iw*main_w/main_h[left_scaled][right_ref];"
        f"[left_scaled][right_ref]hstack"
    )

    cmd = [
        'ffmpeg', '-y',
        '-i', source_video_path,
        '-i', output_video_path,
        '-filter_complex', filter_complex,
        '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
        output_result_path
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"  Saved to {output_result_path}")
    except subprocess.CalledProcessError as e:
        print(f"  Error processing {filename}: {e.stderr.decode()}")

def main():
    files = glob.glob(os.path.join(OUTPUT_DIR, '*.mp4'))
    print(f"Found {len(files)} files in {OUTPUT_DIR}")
    
    for f in files:
        process_video(f)

if __name__ == "__main__":
    main()
