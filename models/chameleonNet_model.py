import torch
from data.new_dataset import NEWDataset
from . import networks
from .networks import ChameleonNet
import torch.nn.functional as F
from torch import nn, cuda
import time
from util import util
import os
import pytorch_lightning as pl
from torchvision.utils import save_image
import torchvision 
import wandb
from pathlib import Path


class DataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
    def setup(self, stage=None):
        self.dataset = NEWDataset(self.opt)

    def train_dataloader(self):
        return(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.opt.batch_size,
                shuffle=True,
                num_workers=int(self.opt.num_threads),
                drop_last=False))

    def test_dataloader(self):
        return(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                shuffle=False,
                num_workers=int(self.opt.num_threads),
                drop_last=False))


class ChameleonNetModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # define networks (both generator and discriminator)
        self.netvgg = networks.vgg
        self.netvgg.load_state_dict(torch.load(opt.vgg))
        self.netvgg = nn.Sequential(*list(self.netvgg.children())[:31])
        if opt.is_skip:
            self.netDecoder = networks.decoder_cat
        else:
            self.netDecoder = networks.decoder
        self.netG = ChameleonNet(self.netvgg, self.netDecoder, is_matting=opt.is_matting, use_dist=opt.use_dist, \
                           is_skip=opt.is_skip, is_fft=opt.is_fft, fft_num=opt.fft_num, split_num=opt.split_num)
        # define loss functions
        self.criterionGAN = nn.MSELoss()
        self.netD = networks.ConvDiscriminator(depth=8, patch_number=opt.patch_number, batchnorm_from=0, use_dist=opt.use_dist)
        self.automatic_optimization = False
        self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=self.opt.lr*self.opt.g_lr_ratio, betas=(self.opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.opt.lr*self.opt.d_lr_ratio, betas=(self.opt.beta1, 0.999))
    
    def configure_optimizers(self):
        return [self.optimizer_G, self.optimizer_D]
        # return self.optimizer_G
    def training_step(self, batch, batch_idx):
        opt_G, opt_D = self.optimizers()

        self.output, self.coarse, self.loss_c, self.loss_s, self.loss_tv = self.netG(batch["img"], batch["mask"], batch["mask_d"])
        self.gan_loss = self.criterionGAN(self.netD(self.output, 0, batch["mask"], batch["mask_d"]), torch.zeros_like(batch["mask_small"]))
        loss_G = self.opt.lambda_content * self.loss_c + self.opt.lambda_style * self.loss_s + self.loss_tv*self.opt.lambda_tv + self.opt.lambda_g * self.gan_loss
        
        # do training_step with decoder
        fake_AB = self.output
        self.pred_fake = self.netD(fake_AB.detach(), 0, batch["mask"], batch["mask_d"]) # output：计算全部
        self.pred_comp = self.netD(batch["img"], 0, batch["mask"], batch["mask_d"]) # comp：计算全部


        output_fake = self.criterionGAN(self.pred_fake, batch["mask_small"])
        composite_fake = self.criterionGAN(self.pred_comp, batch["mask_small"])
        self.loss_D_fake = output_fake + composite_fake
        
        real_AB = batch["img"]
        self.pred_real = self.netD(real_AB, 1, batch["mask"], batch["mask_d"]) 
        self.loss_D_real = self.criterionGAN(self.pred_real, torch.zeros_like(batch["mask_small"]))
        loss_D = self.loss_D_fake + self.loss_D_real

        # Optimize Generator
        opt_G.zero_grad()
        self.manual_backward(loss_G)
        opt_G.step()

        # Optimize Discriminator
        opt_D.zero_grad()
        self.manual_backward(loss_D)
        opt_D.step()
        

    def test_step(self, batch, batch_idx):  
        self.output, _, _, _, _, = self.netG(batch["img"], batch["mask"], batch["mask_d"])
        output = self.output
        outdir = "./gen"

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outdir = os.path.join(outdir, os.path.basename(str(batch["img_path"])))
        outdir = outdir.split("'")[0]
        np_out = util.tensor2im(output)
        util.save_image(np_out, outdir)