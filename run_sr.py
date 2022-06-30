import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary

from generator import *
from losses import *
from datasets import *
from utils import *
from metrics import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import augments
from option import get_option

from flash.core.optimizers import LARS

os.makedirs("train_images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)


cuda = torch.cuda.is_available()

opt = get_option()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator =  Generator(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=20, num_grow_ch=32)

# Losses
criterion_MSE = torch.nn.MSELoss()

device = torch.device("cuda")
if cuda:
    generator = generator.cuda()
    criterion_MSE = criterion_MSE.cuda()
# Load pretrained models
# generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
# discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))


# Optimizers
#optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G =  LARS(
            generator.parameters(),
            opt.lr,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=False,
            eps=1e-8,
            trust_coefficient=0.001
        )

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../sr_datasets/ixi/IXI-T1", hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        imgs_hr, imgs_lr, mask, aug = augments.apply_augment(
                imgs_hr, imgs_lr,
                opt.augs, opt.prob, opt.alpha,
                opt.aux_alpha, opt.aux_alpha, opt.mix_p
            )
        
        trans = transforms.Resize((256//4, 256//4), Image.BICUBIC)
        imgs_lr = trans(imgs_lr)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        loss_pixel = criterion_MSE(gen_hr, imgs_hr)
        # Total loss
        loss_G = loss_pixel
        loss_G.backward()
        optimizer_G.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % 10 == 0:
            # --------------
            #  Log Progress
            # --------------
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_G
                )
            )
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr_cp = tensor2np(gen_hr.detach()[0])
            imgs_hr_cp = tensor2np(imgs_hr.detach()[0])
            psnr = compute_psnr(gen_hr_cp, imgs_hr_cp)
            ssim = compute_ssim(gen_hr_cp, imgs_hr_cp)
            print(psnr,ssim)
            #print("[PSNR %f] [SSIM %f]" % (psnr, ssim))
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, imgs_hr, gen_hr),-1)
            save_image(img_grid, "train_images/%d.png" % batches_done)

    torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)