#!/bin/bash

# Single run inference with out-of-the-box St4RTrack model on DNA_02 sequence
# No TTA - just forward pass

python infer.py \
    --input_dir /home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/data/DNA_Seq/DNA_02 \
    --weights /home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/checkpoints/St4RTrack_Seqmode_reweightMax5.pth \
    --output_dir ./results/DNA_02_baseline \
    --batch_size 48 \
    --image_size 512 \
    --num_frames 200
