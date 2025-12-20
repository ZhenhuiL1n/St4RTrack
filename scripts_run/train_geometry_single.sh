#!/bin/bash

# Fine-tuning script for Geometry Refinement (Single GPU / No Distributed)
# Useful for debugging or simple TTA.

data_root="/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/Part1/0008_01"
ckpt_root="/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/checkpoints"
output_name="DNA_0008_01_GeometryOnly_SingleGPU"

# Use python directly instead of torchrun
# Force CUDA_VISIBLE_DEVICES to a single GPU (e.g., 0)
export CUDA_VISIBLE_DEVICES=0

# Pseudo-distributed setup to satisfy dust3r's barrier() calls
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

python train.py \
--model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 288), \
head_type='dpt', freeze='encoder', output_mode='pts3d', depth_mode=('exp', -inf, inf), \
conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
--train_dataset "300 @ DNADataset(S=12, resolution=[(512, 288)], \
dataset_location='${data_root}')" \
--test_criterion "Regr3D(L21, norm_mode='avg_dis')" \
--pretrained "${ckpt_root}/St4RTrack_Seqmode_reweightMax5.pth" \
--lr 0.00005 --min_lr 4e-05 --warmup_epochs 1 --epochs 35 --batch_size 1 --accum_iter 2 --amp 1 \
--save_freq 5 --keep_freq 5 --pose_eval_freq 20 --fixed_eval_set --num_frames 120 --eval_freq 1 --first_eval \
--tta_eval "${data_root}" \
--train_criterion "ConfLoss(Regr3D(L21, norm_mode=''), alpha=0.2, velo_weight=0, pose_weight=0, \
depth_weight=10.0, traj_weight=0.0, intr_inv_loss=True, pred_intrinsics=True,\
cotracker=False, align3d_weight=5.0)" \
--output_dir "./train_Geometry_results/${output_name}_fixed" \
--world_size 1 \
--num_workers 0 # Set to 0 to enable pdb debugging in main process
