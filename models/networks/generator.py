import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import SPADE, get_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import FADEResnetBlock as FADEResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.architecture import SPADE_FADEResnetBlock as SPADE_FADEResnetBlock
from models.networks.stream import Stream as Stream
from models.networks.AdaIN.function import adaptive_instance_normalization as FAdaIN


class TSITGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralfadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='more',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks."
                                 "If 'most', also add one more upsampling + resnet layer at the end of the generator."
                                 "We only use 'more' as the default setting.")
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        self.content_stream = Stream(self.opt)
        self.style_stream = Stream(self.opt) if not self.opt.no_ss else None
        self.sw, self.sh = self.compute_latent_vector_size(opt)
        str_config = ''
        self.spade1024_1 = SPADE(str_config, 1024, opt.nc_spade_label)
        self.spade1024_2 = SPADE(str_config, 1024, opt.nc_spade_label)
        self.spade1024_3 = SPADE(str_config, 1024, opt.nc_spade_label)
        self.spade1024_4 = SPADE(str_config, 1024, opt.nc_spade_label)
        self.spade512 = SPADE(str_config, 512, opt.nc_spade_label)
        self.spade256 = SPADE(str_config, 256, opt.nc_spade_label)
        self.spade128 = SPADE(str_config, 128, opt.nc_spade_label)
        self.spade64 = SPADE(str_config, 64, opt.nc_spade_label)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map (content) instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADE_FADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADE_FADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADE_FADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADE_FADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADE_FADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADE_FADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADE_FADEResnetBlock(2 * nf, 1 * nf, opt)
        

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADE_FADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 7
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 8
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        
        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)
        return sw, sh

    def fadain_alpha(self, content_feat, style_feat, alpha=1.0, c_mask=None, s_mask=None):
        # FAdaIN performs AdaIN on the multi-scale feature representations
        assert 0 <= alpha <= 1
        t = FAdaIN(content_feat, style_feat, c_mask, s_mask)
        t = alpha * t + (1 - alpha) * content_feat
        return t
    
    def fadain_spade(self, content_feat, style_feat, segmap, spade_module, alpha=1.0, c_mask=None, s_mask=None):
        # FAdaIN performs AdaIN on the multi-scale feature representations
        assert 0 <= alpha <= 1
        t = FAdaIN(content_feat, style_feat, c_mask, s_mask)
        s = spade_module(content_feat, segmap)
        t = alpha * t + (1 - alpha) * content_feat
        return 0.5 * t + 0.5 * s


    def forward(self, input, real, segmap, z=None):
        content = input
        style =  real
        ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7 = self.content_stream(content)
        # print('ataoy', ft0.size(), ft1.size(), ft2.size(), ft3.size(), ft4.size(), ft5.size(), ft6.size(), ft7.size())
        # segft0, segft1, segft2, segft3, segft4, segft5, segft6, segft7 = self.content_stream(segmap)
        sft0, sft1, sft2, sft3, sft4, sft5, sft6, sft7 = self.style_stream(style) if not self.opt.no_ss else [None] * 8
        # print('ataoy', sft0.size(), sft1.size(), sft2.size(), sft3.size(), sft4.size(), sft5.size(), sft6.size(), sft7.size())
        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(content.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=content.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            if self.opt.task == 'SIS':
                # following SPADE, downsample segmap and run convolution for SIS
                x = F.interpolate(content, size=(self.sh, self.sw))
            else:
                # sample random noise
                x = torch.randn(content.size(0), 3, self.sh, self.sw, dtype=torch.float32, device=content.get_device())
            x = self.fc(x)
        x = self.fadain_alpha(x, sft7, alpha=self.opt.alpha) if not self.opt.no_ss else x
        x = self.head_0(x, ft7, segmap)

        #x = self.up(x)
        _,_,h,w = ft6.size()
        x = F.interpolate(x, size=(h, w))
        x = self.fadain_alpha(x, sft6, alpha=self.opt.alpha) if not self.opt.no_ss else x
        x = self.G_middle_0(x, ft6, segmap)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            #x = self.up(x)
            _,_,h,w = ft5.size()
            x = F.interpolate(x, size=(h, w))

        x = self.fadain_alpha(x, sft5, alpha=self.opt.alpha) if not self.opt.no_ss else x
        x = self.G_middle_1(x, ft5, segmap)

        #x = self.up(x)
        _,_,h,w = ft4.size()
        x = F.interpolate(x, size=(h, w))
        x = self.fadain_alpha(x, sft4, alpha=self.opt.alpha) if not self.opt.no_ss else x
        x = self.up_0(x, ft4, segmap)
        #x = self.up(x)
        _,_,h,w = ft3.size()
        x = F.interpolate(x, size=(h, w))
        x = self.fadain_alpha(x, sft3, alpha=self.opt.alpha) if not self.opt.no_ss else x
        x = self.up_1(x, ft3, segmap)
        #x = self.up(x)
        _,_,h,w = ft2.size()
        x = F.interpolate(x, size=(h, w))
        x = self.fadain_alpha(x, sft2, alpha=self.opt.alpha) if not self.opt.no_ss else x
        x = self.up_2(x, ft2, segmap)
        #x = self.up(x)
        _,_,h,w = ft1.size()
        x = F.interpolate(x, size=(h, w))
        x = self.fadain_alpha(x, sft1, alpha=self.opt.alpha) if not self.opt.no_ss else x
        x = self.up_3(x, ft1, segmap)
        #x = self.up(x)
        _,_,h,w = ft0.size()
        x = F.interpolate(x, size=(h, w))
        if self.opt.num_upsampling_layers == 'most':
            ft0 = self.up(ft0)
            x = self.fadain_alpha(x, sft0, alpha=self.opt.alpha) if not self.opt.no_ss else x
            x = self.up_4(x, ft0, segmap)
            x = self.up(x)
        _,_,h,w=content.size()
        x=F.interpolate(x, size=(h,w))
        x = self.conv_img(F.leaky_relu(x, 2e-1))

        x = F.tanh(x)
        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)
