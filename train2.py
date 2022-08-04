import sys
from collections import OrderedDict
from PIL import Image
# from cv2 import threshold

import data
from DeepLabV3Plus_Pytorch.datasets.kitti import Kitti
from DeepLabV3Plus_Pytorch.datasets.vkitti import Vkitti
from options.train_options import TrainOptions
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
from tqdm import tqdm
from DeepLabV3Plus_Pytorch import network, utils, datasets
from torch.utils import data as datatorch
from DeepLabV3Plus_Pytorch.metrics import StreamSegMetrics
from torchmetrics import JaccardIndex
from DeepLabV3Plus_Pytorch.utils import ext_transforms as et
import torch
import torch.nn.functional as F
import os
import torchvision.transforms.functional as tff
from torchvision.utils import save_image, make_grid
import numpy as np
import torchvision.transforms as transforms


def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

def validate(model_seg, model_tsit, kitti_loader, vkitti_loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, ((kitti_images, kitti_labels), (vkitti_images, vkitti_labels)) in tqdm(enumerate(zip(kitti_loader, vkitti_loader))):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            kitti_to_vkitti_image = model_tsit(kitti_images)

            outputs = model_seg(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

        score = metrics.get_results()
    return score, ret_samples

crop_size = 188

# parse options
opt = TrainOptions().parse()
# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
print('DEBUT DATALOADER')
dataloader = data.create_dataloader(opt)
print('FIN DATALOADER')

# create trainer for our model
print('DEBUT NETWWORK')
trainer = Pix2PixTrainer(opt)
seg_model = 0
ckpt='/home/demeter/workspace_remiG/DeepLabV3Plus_Pytorch/saved_checkpoints/best_deeplabv3plus_resnet50_vkitti_os16_best.pth'
seg_step = 100
model_seg = network.modeling.__dict__['deeplabv3plus_resnet50'](num_classes=11, output_stride=16)
# network.convert_to_separable_conv(model_seg.classifier)
utils.set_bn_momentum(model_seg.backbone, momentum=0.01)
device = torch.device('cuda')
print("ISFILE", os.path.isfile(ckpt))
checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
model_seg.load_state_dict(checkpoint["model_state"])
model_seg = torch.nn.DataParallel(model_seg)
model_seg.to(device)
model_seg.eval()
print('FIN NETWORK')

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)


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

vkitti_val_transform = et.ExtCompose([
    # et.ExtResize( 512 ),
    et.ExtResize((188, 620)),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
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

val_ds_kitti = Kitti('DeepLabV3Plus_Pytorch/datasets/data/kitti', split='val', transform=kitti_val_transform)
val_loader_kitti = datatorch.DataLoader(val_ds_kitti, batch_size=1, shuffle=True, num_workers=2)
val_ds_vkitti = Vkitti('DeepLabV3Plus_Pytorch/datasets/data/vkitti', split='val', transform=vkitti_val_transform)
val_loader_vkitti = datatorch.DataLoader(val_ds_vkitti, batch_size=1, shuffle=True, num_workers=2)

def max_with_threshold(seg_pred):
    seg_pred = torch.softmax(seg_pred, dim=1)
    seg_probs, seg_preds = torch.max(seg_pred, dim=1)

def compute_miou(pred, truth):
    inter_list = [0 for _ in range(11)]
    union_list = [0 for _ in range(11)]
    for y in range(pred.shape[1]):
        for x in range(pred.shape[2]):
            p = pred[0, y, x]
            t = truth[0, y, x]
            if p != 255 and t != 255:
                if p == t:
                    union_list[t] += 1
                    inter_list[p] += 1
                else:
                    union_list[t] += 1
                    union_list[p] += 1
    miou_list = [inter_list[i]/union_list[i] if union_list[i] != 0 else np.NaN for i in range(11)]
    return np.nansum(miou_list)/sum(~np.isnan(miou_list)), miou_list

metrics = StreamSegMetrics(11)



metrics.reset()
for (image_kitti, label_kitti), (image_vkitti, label_vkitti) in zip(val_loader_kitti, val_loader_vkitti):
    transform_to_tensor = transforms.Compose([transforms.PILToTensor()])
    print("VOICI TYPE", type(image_kitti), type(image_vkitti))
    image_kitti_val = F.pad(image_kitti, (0, 0, 0, 432, 0, 0, 0, 0), mode='constant', value=0)
    image_vkitti_val = F.pad(image_vkitti, (0, 0, 0, 432, 0, 0, 0, 0), mode='constant', value=0)
    print("VOICI SIZE", image_kitti_val.size(), image_vkitti_val.size())
    data_val = {
        'label' : image_vkitti_val,
        'instance' : 0,
        'image' :image_kitti_val,
        'path' : 0,
        'cpath' : 0
    }
    trainer.run_generator_one_step(data_val)
    input = trainer.get_latest_generated()[0]
    input = torch.unsqueeze(input, 0)
    print('INPUT', input.size())
    seg_pred = model_seg(tff.crop(input, 0, 0, 188, 620))
    output = seg_pred.detach().max(dim=1)[1].cpu().numpy()
    # seg_pred = torch.softmax(seg_pred, dim=1)
    # print('VOICI LE SOFTMAX', seg_pred[0, :, 0, 0], sum(seg_pred[0, :, 0, 0]))
    seg_probs, seg_preds = torch.max(seg_pred, dim=1)
    threshold_seg = 255.0
    jaccard = JaccardIndex(num_classes=11)

    seg_preds_cap = torch.where(seg_probs > 0.9, seg_preds, torch.full(seg_preds.size(), 255).cuda())
    save_image(torch.mul(seg_preds_cap, 1/11), 'wohaha.png')
    metrics.update(label_kitti.cpu().numpy(), output)
    print('MIOU MAISON', compute_miou(output, label_kitti.cpu().numpy()))
    print('SIZE JACCARD', label_kitti.size(), seg_pred.detach().max(dim=1)[1].size())
    #break

print('Training was successfully finished.')
print('VOICI LE MIOU', metrics.get_results())


