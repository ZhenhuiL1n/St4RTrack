#!/bin/bash

# Infer all 48 views for DNA sequence 0012_09
# Uses the TTA-finetuned checkpoint

CHECKPOINT="./train_TTA_results/0012_09_multiview_S8/checkpoint-final.pth"
INPUT_BASE="./data/DNA_infer"
OUTPUT_DIR="./infer_results"
SEQ="0012_09"

for view in $(seq 0 47); do
    echo "========================================"
    echo "Processing view $view of 47"
    echo "========================================"
    
    python infer.py \
        --weights "${CHECKPOINT}" \
        --input_dir "${INPUT_BASE}/${SEQ}_view${view}/" \
        --output_dir "${OUTPUT_DIR}" \
        --seq_name "${SEQ}_view${view}_tta" \
        --batch_size 16 \
        --num_frames 150
done

echo "Done! All 48 views processed."
