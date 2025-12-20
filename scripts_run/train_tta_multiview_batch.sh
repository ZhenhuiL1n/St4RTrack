#!/bin/bash

# TTA with MULTI-VIEW training (simpler approach)
# Uses DNASingleSeqDataset with S=8 views per sample
# batch_size=1 means each iteration processes 8 views from the same timestamp
#
# Each training step:
#   - Picks 1 timestamp (frame) 
#   - Samples 8 random views from that timestamp
#   - Trains on those 8 views together

SEQ_NAME="0012_09"
DATA_DIR="/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/Part1/${SEQ_NAME}"
MOGE_DIR="/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/calibrated_depth"
CKPT_PATH="/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/checkpoints/St4RTrack_Seqmode_reweightMax5.pth"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29642 train.py \
--model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), \
head_type='dpt', freeze='encoder', output_mode='pts3d', depth_mode=('exp', -inf, inf), \
conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
--train_dataset "300 @ DNASingleSeqDataset(dataset_location='${DATA_DIR}', moge_depth_dir='${MOGE_DIR}', \
S=8, resolution=[(512, 512)], view_sample_mode='random', load_moge=True)" \
--test_criterion "Regr3D(L21, norm_mode='avg_dis')" \
--pretrained "${CKPT_PATH}" \
--num_workers 2 \
--lr 0.00005 --min_lr 4e-05 --warmup_epochs 1 --epochs 35 --batch_size 1 --accum_iter 2 --amp 1 --grad_clip \
--save_freq 5 --keep_freq 5 --eval_freq 5 --first_eval \
--train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2, velo_weight=0, pose_weight=0, \
depth_weight=10.0, traj_weight=0.0, intr_inv_loss=False, pred_intrinsics=False, \
cotracker=False, align3d_weight=0, metric_depth_weight=0)" \
--output_dir "./train_TTA_results/${SEQ_NAME}_multiview_S8"
