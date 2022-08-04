NAME=ast_vkitti2kitti_seg_reset_without_G_matched_with_spadefade_bins_one_hotted_k30
TASK=AST 
DATA=custom
CROOT=./datasets/vkitti
SROOT=./datasets/kitti
CKPTROOT=/home/demeter/workspace_remiG/TSIT/checkpoints
WORKER=4
LOADSIZE=620
CROPSIZE=620
ASPECTRATIO=1
PREPROCESS=scale_width_and_crop
WINSIZE=620
BATCHSIZE=1


CUDA_VISIBLE_DEVICES=0 python train3.py --name $NAME \
                 --task $TASK \
                 --gpu_ids 0 \
                 --checkpoints_dir $CKPTROOT \
                 --batchSize $BATCHSIZE \
                 --dataset_mode $DATA \
                 --image_dir $CROOT \
                 --label_dir $SROOT \
                 --segmap_dir ../dataset/kitti_depth \
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
                 --batchSize 1 \
                 --crop_size $CROPSIZE \
                 --aspect_ratio $ASPECTRATIO \
                 --display_winsize $WINSIZE \
                 --nc_spade_label 31 \
                 --nb_bins 30 \
                 --no_pairing_check