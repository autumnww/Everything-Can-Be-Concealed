import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import functools
from torch.nn import init
from torch.optim import lr_scheduler
from models.normalize import RAIN
from torch.nn.utils import spectral_norm
# from function import adaptive_instance_normalization as adain
# from function import calc_mean_std

#from models.unet import *  # added by Joy: U-Net architecture, based on idih

import cv2
import os

class TVLoss(nn.Module):
    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength
        self.x_diff = torch.Tensor()
        self.y_diff = torch.Tensor()

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        # return input
        return self.loss



def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def get_score_map(mask):
    b,_,h,w = mask.shape
    score_map = np.zeros((b, 1, h, w), dtype=np.float32)
    for i in range(b):
        mask_i = (mask[i, 0].cpu().numpy() * 255).astype(np.uint8)   # 转换为 uint8
        distance_transform = cv2.distanceTransform(mask_i, cv2.DIST_L2, 5) # 距离计算
        normalized_distance_transform = cv2.normalize(distance_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        score_map[i, 0] = 1 - normalized_distance_transform
        score_map[i, 0] *= (1 - mask[i, 0].cpu().numpy())
    return score_map

def adain_fg(comp_feat, style_feat, mask, mask_d, use_dist_flag):
    #assert (content_feat.size()[:2] == style_feat.size()[:2])
    num = torch.sum(mask, dim=[2, 3])  
    if (num == 0).all():
        return comp_feat
    size = comp_feat.size()
    style_mean, style_std = get_foreground_mean_std(style_feat, 1-mask_d)
    fore_mean, fore_std = get_foreground_mean_std(comp_feat, mask)  # the foreground features

    normalized_feat = (comp_feat - fore_mean.expand(size)) / fore_std.expand(size)
    return (normalized_feat * style_std.expand(size) + style_mean.expand(size)) * mask + (comp_feat * (1 - mask))


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)), # 0 
    nn.Conv2d(512, 256, (3, 3)), # 1
    nn.ReLU(), # relu1-1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 4
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(), # relu2-1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), # 17
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), # 24
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    # nn.Conv2d(64, 1, (3, 3),padding=0,stride=1), ##matting layer
    nn.Conv2d(65, 1, (1, 1),padding=0,stride=1), ##matting layer 27
    nn.ReflectionPad2d((1, 1, 1, 1)), # 24
    # nn.ReflectionPad2d((1, 1, 1, 1)), ##matting layer
    nn.Conv2d(64, 3, (3, 3)),
)


decoder_cat = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(), # relu1-1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 4
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(), # relu2-1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 17
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 24
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(), # 26

    # nn.Conv2d(65, 1, (1, 1)), ##matting layer 27
    nn.ReflectionPad2d((1, 1, 1, 1)), # 27
    nn.Conv2d(65, 65, (3, 3)),
    nn.ReLU(),
    nn.Conv2d(65, 1, (1, 1)), # 30

    nn.ReflectionPad2d((1, 1, 1, 1)), # 31 dec_2
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),

    nn.ReflectionPad2d((1, 1, 1, 1)), 
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3))
)



vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)), 
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


encoder = nn.Sequential(
    nn.Conv2d(4, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
)



class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    elif norm_type.startswith('rain'):
        norm_layer = RAIN
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
            self.relu = nn.ReLU()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean() # self.relu(1-prediction.mean())
            else:
                loss = prediction.mean() # self.relu(1+prediction.mean())
        return loss


def get_foreground_mean_std(features, mask, eps=1e-5):
    region = features * mask 
    sum = torch.sum(region, dim=[2, 3])     # (B, C)
    num = torch.sum(mask, dim=[2, 3])       # (B, C)
    mu = sum / (num + eps)
    mean = mu[:, :, None, None]
    var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + eps)
    var = var[:, :, None, None]
    std = torch.sqrt(var+eps)
    return mean, std

"""
Generator
"""

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)

def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


class ResBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ResBlock, self).__init__()
        layers = []
        layers += [conv3x3(inplanes, outplanes)]
        layers += [nn.ReLU(inplace=True)]
        layers += [conv3x3(outplanes, outplanes)]

        self.conv = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Conv2d(inplanes, outplanes, (1,1))
    def forward(self, x):
        x_ = self.upsample(x)
        return self.relu(self.conv(x) + x_)
    
def Normalize(in_channels):
    if in_channels<32:
        num_groups = 1
    else:
        num_groups = 32
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ChameleonNet(nn.Module):
    def __init__(self, vgg, decoder, use_dist=True,is_matting=False, is_skip=False, is_fft=False, fft_num=2, split_num=16):
        super(ChameleonNet, self).__init__()
        # load the pretrained VGG encoder
        vgg_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*vgg_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*vgg_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*vgg_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*vgg_layers[18:31])  # relu3_1 -> relu4_1
       
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss(1)

        # define the decoder
        self.decoder = decoder
        dec_layers = list(decoder.children())
        self.dec_1 =  nn.Sequential(*dec_layers[:4]) 
        self.dec_2 =  nn.Sequential(*dec_layers[4:17]) 
        self.dec_3 =  nn.Sequential(*dec_layers[17:24]) 
        self.dec_4 =  nn.Sequential(*dec_layers[24:27]) 
        self.conv_attention = nn.Sequential(*dec_layers[27:31]) 
        self.dec_4_2 =  nn.Sequential(*dec_layers[31:]) 
        self.is_matting = is_matting
        self.is_skip = is_skip
        self.is_fft = is_fft
        self.fft_num = fft_num
        self.split_num = split_num
        self.use_dist = use_dist
        # fix the VGG encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    
    def decode(self, comp, style, mask, enc_features, style_feats):
        width = height = enc_features[-1].size(-1)
        downsample_mask = self.downsample(mask, width, height)
        t = adain_fg(enc_features[-1], style_feats[-1], downsample_mask)
        dec_feature = self.dec_1(t)

        for i in range(1,4):
            func = getattr(self, 'dec_{:d}'.format(i + 1))
            width = height = enc_features[-(i+1)].size(-1)
            downsample_mask = self.downsample(mask, width, height)
            t = adain_fg(enc_features[-(i+1)], style_feats[-(i+1)], downsample_mask)
            dec_feature = func(torch.cat([dec_feature, t], dim=1))

        coarse_output = self.dec_4_2(self.dec_1(dec_feature))
        output = comp * (1-mask) + coarse_output * mask
        return output, coarse_output
    
    def calc_content_loss(self, gen, comb):
        loss = self.mse_loss(gen, comb) 
        return loss
    

    def downsample(self, image_tensor, width, height):
        image_upsample_tensor = torch.nn.functional.interpolate(image_tensor, size=[height, width])
        image_upsample_tensor = image_upsample_tensor.clamp(0, 1)
        return image_upsample_tensor


    def calc_style_loss_mulitple_fg(self, combs, styles, mask, mask_d):
        loss = torch.zeros(1).cuda()
        for i in range(0, 4):
            width = height = combs[i].size(-1)
            downsample_mask = self.downsample(mask, width, height)
            #downsample_mask_style = torch.ones(downsample_mask.size()).cuda()
            downsample_mask_d = self.downsample(mask_d, width, height)
            mu_cs,sigma_cs = get_foreground_mean_std(combs[i], downsample_mask)
            mu_target,sigma_target = get_foreground_mean_std(styles[i], 1-downsample_mask_d)
            loss_i = self.mse_loss(mu_cs, mu_target) + self.mse_loss(sigma_cs, sigma_target)
            loss += loss_i
        return loss

    def fun(self, mask, mask_d, img_feates):
        styled_feature = []
        for i in range(4):
            height = img_feates[i].size(-2)
            width = img_feates[i].size(-1)
            downsample_mask = self.downsample(mask, width, height)
            downsample_mask_d = self.downsample(mask_d, width, height)
            t = adain_fg(img_feates[i], img_feates[i], downsample_mask, downsample_mask_d, self.use_dist)
            styled_feature.append(t)
        return styled_feature

    def fun_2(self, dec_feature, img, mask):
        t = dec_feature[-1]
        dec = self.dec_1(t)
        dec = self.dec_2(torch.concat([dec, dec_feature[-2]], dim=1))
        dec = self.dec_3(torch.concat([dec, dec_feature[-3]], dim=1))
        dec = self.dec_4(torch.concat([dec, dec_feature[-4]], dim=1))

        coarse_output = self.dec_4_2(dec)
        output = img * (1-mask) + coarse_output * mask
        return output, coarse_output

    def split_2(self, img, mask, mask_d): # 5. 输入mask_d
        split_num = self.split_num
        sliced_img = torch.chunk(img, split_num, dim=-2)
        sliced_mask = torch.chunk(mask, split_num, dim=-2)
        sliced_mask_d = torch.chunk(mask_d, split_num, dim=-2)

        styled_feats = []
        for i in range(split_num):
            img_feats = self.encode_with_intermediate(sliced_img[i])
            feat =self.fun(sliced_mask[i], sliced_mask_d[i], img_feats)
            styled_feats.append(feat)

        # feat = torch.cat(feat, dim=-2)
        feat = []
        for i in range(4):
            dec = torch.concat([styled_feats[0][i], styled_feats[1][i]], dim=-2)
            for j in range(2, split_num):
                dec = torch.concat([dec, styled_feats[j][i]], dim=-2)
            feat.append(dec)
        
        output, coarse_output = self.fun_2(feat, img, mask)

        return output, coarse_output

    def cam(self, img, mask, mask_d):
        split_num = self.split_num
        img_feats = self.encode_with_intermediate(img)
        styled_feats =self.fun(mask, mask_d, img_feats)
    
        output, coarse_output = self.fun_2(styled_feats, img, mask)

        return output, coarse_output

    def forward(self, img, mask, mask_d): # 1. 输入 mask_d
        # if split_num != 1
        if self.split_num == 1:
            final_output, coarse_output = self.cam(img, mask, mask_d)
        else:
            final_output, coarse_output = self.split_2(img, mask, mask_d) # 2. 输入 mask_d
        coarse_feats = self.encode_with_intermediate(coarse_output)
        fine_feats = self.encode_with_intermediate(final_output)
        img_feats = self.encode_with_intermediate(img)
        # calculate content loss
        loss_c = self.calc_content_loss(coarse_feats[-1], img_feats[-1])
        loss_c += self.calc_content_loss(fine_feats[-1], img_feats[-1])
        # calculate style loss
        loss_s = self.calc_style_loss_mulitple_fg(coarse_feats, img_feats, mask, mask_d) # 3. 输入mask_d
        loss_s += self.calc_style_loss_mulitple_fg(fine_feats, img_feats, mask, mask_d) # 4. 输入mask_d
        # calculate smooth loss
        tv_loss = self.tv_loss(final_output)

        return final_output, coarse_output, loss_c, loss_s, tv_loss

"""
Discriminator
"""

class FFTBlock_D(nn.Module):
    def __init__(self, depth, in_channels, ch, output_channels, norm_layer=nn.BatchNorm2d, batchnorm_from=0, max_channels=512, fft_norm='ortho'):
        super(FFTBlock_D, self).__init__()
        self.fft_norm = fft_norm
        self.depth = depth
        self.use_fc = False
        in_channels = in_channels
        out_channels = ch

        self.blocks_connected = nn.ModuleDict()
        for block_i in range(0, depth):
            if block_i % 2:
                #in_channels = out_channels
                in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
            else:
                #in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
                in_channels = out_channels
            #print(in_channels, out_channels)
            self.blocks_connected[f'block{block_i}'] = ConvBlock_D(
                in_channels, out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                padding=int(block_i < depth - 1)
            )
        if out_channels > output_channels:
            self.fc = nn.Conv2d(max_channels, output_channels, kernel_size=1)
            self.use_fc = True


    def forward(self, x):
        batch = x.shape[0]

        r_size = x.size()
        fft_dim = (-2, -1)
        x_fft = torch.fft.fftn(x, dim=fft_dim, norm=self.fft_norm)
        x_fft = torch.stack((x_fft.real, x_fft.imag), dim=-1)
        x_fft = x_fft.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w)
        x_fft = x_fft.view((batch, -1,) + x_fft.size()[3:])  # (batch, c * 2, h, w)

        for block_i in range(0, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            x_fft = block(x_fft)
        if self.use_fc:
            output = self.fc(x_fft)
        else:
            output = x_fft

        return output


class ConvBlock_D(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size=4, stride=2, padding=1,
        norm_layer=nn.BatchNorm2d, activation=nn.ELU,
        bias=True,
    ):
        super(ConvBlock_D, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            nn.LeakyReLU(0.2, True),
        )
    def forward(self, x):
        return self.block(x)


class ConvEncoder_D(nn.Module):
    def __init__(
        self,
        depth, ch, patch_number,
        norm_layer, batchnorm_from, max_channels, use_dist
    ):
        super(ConvEncoder_D, self).__init__()
        self.depth = depth
        self.patch_number = patch_number
        self.use_dist = use_dist
        in_channels = 3
        out_channels = ch

        self.block0 = ConvBlock_D(in_channels, out_channels, norm_layer=norm_layer if batchnorm_from == 0 else None)
        self.block1 = ConvBlock_D(out_channels, out_channels, norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None)
        self.blocks_enc = nn.ModuleDict()
        for block_i in range(2, depth-2):
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
            self.blocks_enc[f'block{block_i}'] = ConvBlock_D(
                in_channels, out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                padding=1 #int(block_i < depth - 1)
            )

        self.blocks_connected = nn.ModuleDict()
    
        for block_i in range(depth - 2, depth):
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
            self.blocks_connected[f'block{block_i}'] = ConvBlock_D(
                in_channels, out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                kernel_size=3, stride=1, padding=int(block_i < depth - 1)
            )
        self.inner_channels = out_channels
        
    def forward(self, x, sign, mask, mask_d=None):
        x = self.block0(x)
        x = self.block1(x)

        size = x.size()
        width = x.size()[-1] 
        height = x.size()[-2]
        downsample_mask = F.interpolate(mask, [height, width])
        downsample_mask = downsample_mask.clamp(0, 1) # 限制在0-1
        if sign:
            if self.use_dist:
                downsample_mask_d = F.interpolate(mask_d, [height, width])
                score_map_d = get_score_map(downsample_mask_d).astype(np.float32)
                score_map_d = torch.from_numpy(score_map_d).to(x.device)

                d_size = score_map_d.size()
                score_map_d = F.softmax(score_map_d.view(score_map_d.size()[0], -1), dim=-1)
                score_map_d = score_map_d.reshape(d_size)

                mean_feature = torch.sum(score_map_d * x, axis=(2,3), keepdim=True)
                x = mean_feature.expand(size) * downsample_mask + x * (1-downsample_mask)
            else:
                mean, std = get_foreground_mean_std(x, 1-downsample_mask)
                x_ = x * std.expand(size) + mean.expand(size)
                x = x_ * downsample_mask + x * (1-downsample_mask)

        for block_i in range(2, self.depth - 2):
            block = self.blocks_enc[f'block{block_i}']
            x = block(x)
       
        output = x

        for block_i in range(self.depth - 2, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            output = block(output)
        return output


class DeconvDecoder_D(nn.Module):
    def __init__(self, depth, encoder_innner_channels, norm_layer):
        super(DeconvDecoder_D, self).__init__()
        self.deconv_blocks = nn.ModuleList()

        in_channels = encoder_innner_channels
        self.deconv_block0 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
            norm_layer(in_channels // 2) if norm_layer is not None else nn.Identity(),
            nn.ReLU(True),
        )
        self.deconv_block1 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=1, padding=1),
            norm_layer(in_channels // 4) if norm_layer is not None else nn.Identity(),
            nn.ReLU(True),
        )

        self.to_binary = nn.Conv2d(in_channels // 4, 1, kernel_size=1)

    def forward(self, encoder_outputs, image):

        output = self.deconv_block0(encoder_outputs)
        output = self.deconv_block1(output)

        output = self.to_binary(output)

        return output

class ConvDiscriminator(nn.Module):
    def __init__(
        self,
        depth, patch_number,
        norm_layer=nn.BatchNorm2d, batchnorm_from=0,
        ch=64, max_channels=512, use_dist=True
    ):
        super(ConvDiscriminator, self).__init__()
        self.depth = depth
        self.patch_number = patch_number
        self.encoder = ConvEncoder_D(depth, ch, patch_number, norm_layer, batchnorm_from, max_channels, use_dist)
        self.decoder = DeconvDecoder_D(2, self.encoder.inner_channels, norm_layer)

    def forward(self, image, sign, mask, mask_d=None):
        intermediates = self.encoder(image, sign, mask, mask_d)
        output = self.decoder(intermediates, image)
        return output