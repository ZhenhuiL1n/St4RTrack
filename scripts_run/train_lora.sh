#!/bin/bash

# This script is used to train the model with LoRA fine-tuning.

# Set data path
your_data_path="/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/data/DNA_Seq"
your_ckpt_path="/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/checkpoints"
seq_name="DNA_01"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29636 train.py \
--model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 288), \
head_type='dpt', freeze='encoder', output_mode='pts3d', depth_mode=('exp', -inf, inf), \
conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, \
lora_r=4, lora_alpha=4.0)" \
--train_dataset "300 @ CustomDUSt3R(S=12, resolution=[(512, 288)], \
dataset_location='${your_data_path}/${seq_name}', \
depth_path='${your_data_path}/depth_maps/${seq_name}')" \
--test_criterion "Regr3D(L21, norm_mode='avg_dis')" \
--pretrained "${your_ckpt_path}/St4RTrack_Seqmode_reweightMax5.pth" \
--lr 0.00005 --min_lr 4e-05 --warmup_epochs 1 --epochs 35 --batch_size 1 --accum_iter 2 --amp 1 \
--save_freq 5 --keep_freq 5 --pose_eval_freq 20 --fixed_eval_set --num_frames 120 --eval_freq 1 --first_eval \
--tta_eval "${your_data_path}/${seq_name}" \
--train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2, velo_weight=1, pose_weight=0, \
depth_weight=10.0, traj_weight=0.5, intr_inv_loss=True, pred_intrinsics=True,\
cotracker=True, align3d_weight=5.0)" \
--output_dir "./train_LoRA_results/${seq_name}"
