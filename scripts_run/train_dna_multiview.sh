#!/bin/bash

# Train St4RTrack on DNA-Rendering toy_dataset with GT depth supervision.
# Uses ALL sequences (35 sequences after cleanup) and ALL views (48 per frame).

# Paths
DATA_DIR="/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/Part1"
CKPT_PATH="/home/longnhat/Lin_workspace/8TB2/Lin/PhDprojects/Sotaas/St4RTrack/checkpoints/St4RTrack_Seqmode_reweightMax5.pth"

# Sequences to exclude (incomplete depth_rendered)
EXCLUDE_SEQS="0018_05,0079_02,0115_07,0239_01,0309_03"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29640 train.py \
--model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), \
head_type='dpt', freeze='encoder', output_mode='pts3d', depth_mode=('exp', -inf, inf), \
conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
--train_dataset "300 @ DNAMultiSeqDataset(dataset_location='${DATA_DIR}', \
moge_depth_dir='/home/longnhat/Lin_workspace/8TB2/Lin/nas-train/toy_dataset/calibrated_depth', \
S=8, resolution=[(512, 512)], stride=1, view_sample_mode='random', load_moge=False, \
exclude_seqs=['0018_05','0079_02','0115_07','0239_01','0309_03'])" \
--test_criterion "Regr3D(L21, norm_mode='avg_dis')" \
--pretrained "${CKPT_PATH}" \
--num_workers 2 \
--lr 0.00005 --min_lr 4e-05 --warmup_epochs 2 --epochs 50 --batch_size 1 --accum_iter 4 --amp 1 --grad_clip \
--save_freq 5 --keep_freq 10 --eval_freq 5 --first_eval \
--train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2, velo_weight=0, pose_weight=0, \
depth_weight=10.0, traj_weight=0.0, intr_inv_loss=True, pred_intrinsics=True, \
cotracker=False, align3d_weight=0, metric_depth_weight=0, bg_depth_weight=0)" \
--output_dir "./train_results/dna_multiview_gt_depth"
