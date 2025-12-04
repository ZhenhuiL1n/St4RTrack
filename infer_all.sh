#!/usr/bin/env bash
set -e

# ====== CONFIGURE THESE PATHS ======
WEIGHTS="/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/checkpoints/St4RTrack_Seqmode_reweightMax5.pth"
INPUT_ROOT="/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/eval_benchmark/lab_Seq"
OUT_ROOT="results"
BATCH_SIZE=48
# ===================================

mkdir -p "$OUT_ROOT"

# Loop over all .mp4 files in the input folder
for video in "$INPUT_ROOT"/*.mp4; do
    # Skip if no files match
    [ -e "$video" ] || continue

    base=$(basename "$video")       # e.g. DNA_random.mp4
    name="${base%.*}"              # e.g. DNA_random
    out_dir="${OUT_ROOT}/${name}"  # e.g. results/DNA_random

    mkdir -p "$out_dir"

    echo "Processing: $video"
    echo "Output dir: $out_dir"

    python infer.py \
        --input_dir "$video" \
        --weights "$WEIGHTS" \
        --output_dir "$out_dir" \
        --batch_size "$BATCH_SIZE"
done

echo "All sequences processed."
