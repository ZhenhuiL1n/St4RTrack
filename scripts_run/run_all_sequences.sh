    #!/bin/bash
# Process all sequences in toy_dataset/Part1
# Runs 48 views per sequence in parallel on 7 GPUs
# Saves to toy_dataset/calibrated_depth/{seq_id}/

set -e

# Configuration
DATA_DIR="/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/Part1"
OUTPUT_DIR="/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/calibrated_depth"
SCRIPT_DIR="/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack"

# GPU settings
GPUS="0,1,2,3,4,5,6"
WORKERS=7

# Activate conda
source /home/longnhat/miniconda3/etc/profile.d/conda.sh
conda activate st4rtrack

cd $SCRIPT_DIR

# Create output directory
mkdir -p $OUTPUT_DIR

# Get all sequence IDs
SEQUENCES=(
    "0008_01"
    "0012_09"
    "0018_05"
    "0019_06"
    "0022_10"
    "0025_11"
    "0031_03"
    "0034_04"
    "0047_01"
    "0047_12"
    "0079_02"
    "0094_02"
    "0095_01"
    "0097_04"
    "0102_02"
    "0111_08"
    "0113_06"
    "0115_07"
    "0118_07"
    "0121_02"
    "0123_02"
    "0124_03"
    "0128_04"
    "0133_07"
    "0147_04"
    "0152_01"
    "0165_08"
    "0166_04"
    "0174_09"
    "0188_02"
    "0196_09"
    "0206_04"
    "0219_07"
    "0235_11"
    "0239_01"
    "0241_10"
    "0307_03"
    "0307_07"
    "0309_03"
    "0310_04"
)

TOTAL=${#SEQUENCES[@]}
echo "============================================================"
echo "Processing $TOTAL sequences"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $GPUS ($WORKERS workers)"
echo "============================================================"

for i in "${!SEQUENCES[@]}"; do
    SEQ="${SEQUENCES[$i]}"
    SEQ_OUTPUT="$OUTPUT_DIR/$SEQ"
    
    echo ""
    echo "============================================================"
    echo "[$((i+1))/$TOTAL] Processing sequence: $SEQ"
    echo "============================================================"
    
    python scripts_run/process_sequence.py \
        --seq "$SEQ" \
        --views all \
        --fast \
        --workers $WORKERS \
        --gpus "$GPUS" \
        --output_dir "$SEQ_OUTPUT"
    
    echo "✓ Completed: $SEQ"
done

echo ""
echo "============================================================"
echo "ALL DONE! Processed $TOTAL sequences"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
