#!/usr/bin/env bash

set -x

NAME='ast_vkitti2kitti_lr_by_10'
TASK='AST'
DATA='custom'
CROOT='../dataset/kitti2vkitti_depth/testB'
SROOT='../dataset/kitti2vkitti_depth/testA'
CKPTROOT='./checkpoints'
WORKER=4
RESROOT='./results/depth'
EPOCH=$1
PREPROCESS='scale_width_and_crop'
LOADSIZE=620
CROPSIZE=620
ASPECTRATIO=1


python test.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 2 \
    --checkpoints_dir $CKPTROOT \
    --batchSize 1 \
    --dataset_mode $DATA \
    --image_dir $CROOT \
    --label_dir $SROOT \
    --load_size $LOADSIZE \
    --crop_size $CROPSIZE \
    --nThreads $WORKER \
    --no_pairing_check \
    --no_instance \
    --num_upsampling_layers more \
    --alpha 1.0 \
    --results_dir $RESROOT$1 \
    --preprocess_mode $PREPROCESS \
    --which_epoch $EPOCH \
    --show_input \
    --aspect_ratio $ASPECTRATIO
