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

os.makedirs("test_images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.2, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()

cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator =  Generator(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=20, num_grow_ch=32)

# Load pretrained models
model_dict = "saved_models/generator_46.pth"
generator.load_state_dict(torch.load(model_dict))

device = torch.device("cuda")
if cuda:
    generator = generator.cuda()

# Load pretrained models
# generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
# discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../sr_datasets/ixi/test", hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Testing
# ----------
total_psnr = 0
total_ssim = 0
for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
        gen_hr_cp = tensor2np(gen_hr.detach()[0])
        imgs_hr_cp = tensor2np(imgs_hr.detach()[0])
        psnr = compute_psnr(gen_hr_cp, imgs_hr_cp)
        ssim = compute_ssim(gen_hr_cp, imgs_hr_cp)
        total_psnr += psnr
        #total_ssim += ssim
        print(psnr,ssim)
        #print("[PSNR %f] [SSIM %f]" % (psnr, ssim))
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
        imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
        img_grid = torch.cat((imgs_lr, imgs_hr, gen_hr),-1)
        save_image(img_grid, "test_images/%d.png" % i)

print(total_psnr)