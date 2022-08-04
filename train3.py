import sys
from collections import OrderedDict
from PIL import Image
# from cv2 import threshold

import data
from data.base_dataset import BaseDataset, get_params, get_transform
from DeepLabV3Plus_Pytorch.datasets.kitti import Kitti
from DeepLabV3Plus_Pytorch.datasets.vkitti import Vkitti
from options.train_options import TrainOptions
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer_seg import Pix2PixTrainer
from tqdm import tqdm
from DeepLabV3Plus_Pytorch import network, utils, datasets
from torch.utils import data as datatorch
from DeepLabV3Plus_Pytorch.metrics import StreamSegMetrics
from DeepLabV3Plus_Pytorch.utils import ext_transforms as et
from skimage.exposure import match_histograms
import torch
import torch.nn.functional as F
import os
import torchvision.transforms.functional as tff
from torchvision.utils import save_image, make_grid
import numpy as np
import torchvision.transforms as transforms



def save_labels(pseudo, preds, epoch, iter, name):
    pseudo_arr = (torch.squeeze(pseudo).cpu().numpy().astype('uint8'))*23
    preds_arr = (torch.squeeze(preds).cpu().numpy().astype('uint8'))*23
    image_arr = np.concatenate((pseudo_arr, np.full((30, 620), 255, dtype=np.uint8), preds_arr), axis=0)
    image = Image.fromarray(image_arr, mode='L')
    image.save('checkpoints/{}/seg_preds/epoch{}_iter{}.png'.format(name, epoch, iter))


crop_size = 188
best_miou = 0.0
best_miou_matched = 0.0

# parse options
opt = TrainOptions().parse()
# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# Create the Semantic Segmentation Model (DeeplabV3Plus)
seg_model = 0
ckpt='/home/demeter/workspace_remiG/DeepLabV3Plus_Pytorch/saved_checkpoints/best_deeplabv3plus_resnet50_vkitti_os16_best.pth'
seg_step = 200
model_seg = network.modeling.__dict__['deeplabv3plus_resnet50'](num_classes=11, output_stride=16)
utils.set_bn_momentum(model_seg.backbone, momentum=0.01)
device = torch.device('cuda')
checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
model_seg.load_state_dict(checkpoint["model_state"])
model_seg = torch.nn.DataParallel(model_seg)
model_seg.to(device)
model_seg.eval()

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

# Transforms
vkitti_train_transform = et.ExtCompose([
    # et.ExtResize( 512 ),
    et.ExtResize((188, 620)),
    et.ExtRandomCrop(size=(crop_size, crop_size)),
    et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    et.ExtRandomHorizontalFlip(),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])



vkitti_val_transform_deeplab = et.ExtCompose([
    # et.ExtResize( 512 ),
    #et.ExtResize((188, 620)),
    #et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])


vkitti_val_transform_TSIT = et.ExtCompose([
    # et.ExtResize( 512 ),
    et.ExtResize((188, 620)),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),
])



kitti_train_transform = et.ExtCompose([
    # et.ExtResize( 512 ),
    et.ExtResize((188, 620)),
    et.ExtRandomCrop(size=(crop_size, crop_size)),
    et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    et.ExtRandomHorizontalFlip(),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

kitti_val_transform = et.ExtCompose([
    # et.ExtResize( 512 ),
    et.ExtResize((188, 620)),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

transform_depth_val = get_transform(opt, {'crop_pos': (0, 0), 'flip': False}, normalize=False, toTensor=False, toPILTensor=True)

val_ds_kitti = Kitti('./datasets/kitti_depth', split='val', transform=vkitti_val_transform_TSIT, depth=True)
val_loader_kitti = datatorch.DataLoader(val_ds_kitti, batch_size=1, shuffle=True, num_workers=2)
val_ds_vkitti = Vkitti('DeepLabV3Plus_Pytorch/datasets/data/vkitti', split='val', transform=vkitti_val_transform_TSIT, keep_all_id=True)
val_loader_vkitti = datatorch.DataLoader(val_ds_vkitti, batch_size=1, shuffle=True, num_workers=2)

metrics = StreamSegMetrics(11)
metrics_matched = StreamSegMetrics(11)


for epoch in tqdm(iter_counter.training_epochs()):
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(tqdm(dataloader), start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i, model_seg, vkitti_val_transform_deeplab)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying() or i==0:
            if opt.task == 'SIS':
                visuals = OrderedDict([('input_label', data_i['label'][0]),
                                       ('synthesized_image', trainer.get_latest_generated()[0]),
                                       ('real_image', data_i['image'][0])])
            else:
                visuals = OrderedDict([('content', data_i['label'][0]),
                                       ('synthesized_image', trainer.get_latest_generated()[0]),
                                       ('style', data_i['image'][0])])
                visuals_seg = OrderedDict([('seg_pseudo', trainer.seg_pseudo),
                                            ('seg_pred', trainer.seg_pred)])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)
            save_labels(trainer.seg_pseudo, trainer.seg_pred, epoch, iter_counter.total_steps_so_far, opt.name)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()
        if i % seg_step == 0:
            metrics.reset()
            metrics_matched.reset()
            for (image_kitti, label_kitti, depth_kitti_path), (image_vkitti, label_vkitti) in zip(val_loader_kitti, val_loader_vkitti):
                depth_kitti = Image.open(depth_kitti_path[0])
                image_kitti_val = F.pad(image_kitti, (0, 0, 0, 432, 0, 0, 0, 0), mode='constant', value=-1.0)
                image_vkitti_val = F.pad(image_vkitti, (0, 0, 0, 432, 0, 0, 0, 0), mode='constant', value=-1.0)

                depth_kitti_val = transform_depth_val(depth_kitti)
                depth_kitti_val = torch.unsqueeze(depth_kitti_val, 0)
                label_vkitti_val = F.pad(torch.unsqueeze(label_vkitti, 0), (0, 0, 0, 432, 0, 0, 0, 0), mode='constant', value=0).float()

                min_tensor = depth_kitti_val.min()
                depth_kitti_val = depth_kitti_val.add(-min_tensor + 1.0)
                q = torch.log(depth_kitti_val.max())/opt.nb_bins
                depth_kitti_val = depth_kitti_val.log().mul(1/q).round()

                segmap_tensor_remi_transform = torch.nn.functional.one_hot(depth_kitti_val.to(torch.int64), opt.nb_bins+1)
                segmap_tensor_remi_transform = torch.squeeze(segmap_tensor_remi_transform)
                segmap_tensor_remi_transform = torch.permute(segmap_tensor_remi_transform, (2, 0, 1)).to(torch.float)
                segmap_tensor_remi_transform = torch.unsqueeze(segmap_tensor_remi_transform, 0)

                data_val = {
                    'label' : image_kitti_val,
                    'instance' : 0,
                    'segmap' :  segmap_tensor_remi_transform,
                    'image' :image_vkitti_val,
                    'path' : 0,
                    'cpath' : 0
                }

                # get transformed image from TSIT
                _, input = trainer.pix2pix_model(data_val, mode='generator')

                # data preparation for DeepLab
                input=input[0].mul(255).add_(0.5).clamp_(0, 255)/255
                input,_=vkitti_val_transform_deeplab(input,label_kitti)
                input = torch.unsqueeze(input, 0)
                input_array = tff.crop(input, 0, 0, 188, 620).detach().cpu().numpy()
                input_matched = torch.from_numpy(match_histograms(input_array, image_vkitti.cpu().numpy())).float()

                # get outputs from DeepLab
                seg_pred = model_seg(tff.crop(input, 0, 0, 188, 620))
                seg_pred_matched = model_seg(input_matched)
                output = seg_pred.detach().max(dim=1)[1].cpu().numpy()
                output_matched = seg_pred_matched.detach().max(dim=1)[1].cpu().numpy()
                seg_probs, seg_preds = torch.max(seg_pred, dim=1)
                threshold_seg = 255.0

                seg_preds_cap = torch.where(seg_probs > 0.9, seg_preds, torch.full(seg_preds.size(), 255).cuda())
                metrics.update(label_kitti.cpu().numpy(), output)
                metrics_matched.update(label_kitti.cpu().numpy(), output_matched)
                print('Scores : ', metrics.get_results())
                print('Scores with histogram matching', metrics_matched.get_results())
            print('Final scores', metrics.get_results())
            print('Final scores with histogram matching', metrics_matched.get_results())

            # saving the best models
            if metrics.get_results()['Mean IoU'] > best_miou:
                best_miou = metrics.get_results()['Mean IoU']
                trainer.save('best_without_match_spadefade_bins_one_hotted_k30')
            if metrics_matched.get_results()['Mean IoU'] > best_miou_matched:
                best_miou_matched = metrics_matched.get_results()['Mean IoU']
                trainer.save('best_with_match_spadefade_bins_one_hotted_k30')
            print('Best scores (without hist matching, with hist matching)', best_miou, best_miou_matched)
            with open('miou_without_G_matched_spadefade_bins_one_hotted_k30.log','a') as mioulog:
                mioulog.write('{}     {}   \n'.format(metrics.get_results()['Mean IoU'], metrics_matched.get_results()['Mean IoU']))


    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
