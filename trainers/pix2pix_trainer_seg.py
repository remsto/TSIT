from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
from DeepLabV3Plus_Pytorch import network, utils
import torch
import torchvision.transforms.functional as tff

criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
alpha = 1.0

class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_generator_one_step(self, data, model_seg, transform_seg):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.pix2pix_model(data, mode='generator')


        input = generated[0].mul(255).add_(0.5).clamp_(0, 255)/255
        input, _ = transform_seg(input, None)
        input = torch.unsqueeze(input, 0)
        seg_output = model_seg(tff.crop(data['label'], 0, 0, 188, 620))
        seg_output = torch.softmax(seg_output, dim=1)
        seg_probs, seg_preds = torch.max(seg_output, dim=1)
        seg_pseudo = torch.where(seg_probs > 0.9, seg_preds, torch.full(seg_preds.size(), 255).cuda())
        seg_pred = model_seg(tff.crop(input, 0, 0, 188, 620))
        seg_loss = criterion(seg_pred, seg_pseudo)
        with open('seg_loss_reset_without_G_matched_spadefade_bins_one_hotted_k30.log','a') as seg_losslog:
            seg_losslog.write(str(seg_loss.item())+'\n')
        g_loss = (1-alpha)*(sum(g_losses.values()).mean()) + alpha*seg_loss
        g_loss.backward()
        self.optimizer_G.step()
        self.seg_pseudo = seg_pseudo
        _, self.seg_pred = torch.max(seg_pred, dim=1)
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
