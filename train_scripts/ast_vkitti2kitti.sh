#!/usr/bin/env bash

set -x

NAME='ast_vkitti2kitti_seg'
TASK='AST'
DATA='custom'
CROOT='./datasets/vkitti'
SROOT='./datasets/kitti'
CKPTROOT='./checkpoints'
WORKER=4
LOADSIZE=620
CROPSIZE=620
ASPECTRATIO=1
PREPROCESS='scale_width_and_crop'
WINSIZE=620
BATCHSIZE=2

CUDA_VISIBLE_DEVICES=3 python train.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 0 \
    --checkpoints_dir $CKPTROOT \
    --batchSize 2 \
    --dataset_mode $DATA \
    --image_dir $CROOT \
    --label_dir $SROOT \
    --nThreads $WORKER \
    --no_pairing_check \
    --no_instance \
    --gan_mode hinge \
    --num_upsampling_layers more \
    --alpha 1.0 \
    --display_freq 200 \
    --save_epoch_freq 20 \
    --niter 200 \
    --lambda_vgg 1 \
    --lambda_feat 1 \
    --load_size $LOADSIZE \
    --preprocess_mode $PREPROCESS \
    --batchSize $BATCHSIZE \
    --crop_size $CROPSIZE \
    --aspect_ratio $ASPECTRATIO \
    --display_winsize $WINSIZE \
    --no_pairing_check \
    --lr 0.00002
    # --tf_log \
    # --ngf 32