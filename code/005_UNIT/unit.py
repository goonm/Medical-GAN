import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import random
import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from batchgenerators.utilities.file_and_folder_operations import *


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--n_itr", type=int, default=25, help="number of iterations")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
opt = parser.parse_args()
print(opt)

# Create result directories
os.makedirs("result", exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_pixel = torch.nn.L1Loss()

device = torch.device('cuda:0')
print('torch.cuda: ',torch.cuda.is_available())

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Dimensionality (channel-wise) of image embedding
shared_dim = opt.dim * 2 ** opt.n_downsample

# Initialize generator and discriminator
shared_E = ResidualBlock(features=shared_dim)
E1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
E2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
shared_G = ResidualBlock(features=shared_dim)
G1 = Generator(dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)
G2 = Generator(dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)
D1 = Discriminator(input_shape)
D2 = Discriminator(input_shape)

# to cuda
E1.to(device)
E2.to(device)
G1.to(device)
G2.to(device)
D1.to(device)
D2.to(device)
criterion_GAN.to(device)
criterion_pixel.to(device)

# Initialize weights
E1.apply(weights_init_normal)
E2.apply(weights_init_normal)
G1.apply(weights_init_normal)
G2.apply(weights_init_normal)
D1.apply(weights_init_normal)
D2.apply(weights_init_normal)

# Loss weights
lambda_0 = 10  # GAN
lambda_1 = 0.1  # KL (encoded images)
lambda_2 = 100  # ID pixel-wise
lambda_3 = 0.1  # KL (encoded translated images)
lambda_4 = 100  # Cycle pixel-wise

# Optimizers
params = list(E1.parameters()) + list(E2.parameters()) + list(G1.parameters()) + list(G2.parameters())
optimizer_G = torch.optim.Adam(
    params,
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor

# Image transformations
transforms_ = [
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
]
val_transforms_ = [
    transforms.ToTensor(),
]

# Configure data loader
data_path = subfiles('data/train', suffix='.nii.gz')
data_path_val = subfiles('data/val', suffix='.nii.gz')

train_dataset = CustomDataset(data_path, opt.img_height, transforms_, mode='train')
val_dataset = CustomDataset(data_path_val, opt.img_height, val_transforms_, mode='test')

dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)

def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss

# ----------
#  Training
# ----------

t0 = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    t1 = time.time()
    loss_G_epoch=[]
    loss_D1_epoch=[]
    loss_D2_epoch=[]
    
    for i, batch in enumerate(dataloader):
        # Set model input
        X1 = Variable(batch["A"].type(Tensor))
        X2 = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths (input shape // 16)
        valid = Variable(Tensor(np.ones((X1.size(0), *D1.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((X1.size(0), *D1.output_shape))), requires_grad=False)
        
        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------
        E1.train()
        E2.train()
        G1.train()
        G2.train()

        optimizer_G.zero_grad()

        # Get shared latent representation
        mu1, Z1 = E1(X1)
        mu2, Z2 = E2(X2)

        # Reconstruct images
        recon_X1 = G1(Z1)
        recon_X2 = G2(Z2)

        # Translate images
        fake_X1 = G1(Z2)
        fake_X2 = G2(Z1)

        # Cycle translation
        mu1_, Z1_ = E1(fake_X1)
        mu2_, Z2_ = E2(fake_X2)
        cycle_X1 = G1(Z2_)
        cycle_X2 = G2(Z1_)

        # Losses
        loss_GAN_1 = lambda_0 * criterion_GAN(D1(fake_X1), valid)
        loss_GAN_2 = lambda_0 * criterion_GAN(D2(fake_X2), valid)
        loss_KL_1 = lambda_1 * compute_kl(mu1)
        loss_KL_2 = lambda_1 * compute_kl(mu2)
        loss_ID_1 = lambda_2 * criterion_pixel(recon_X1, X1)
        loss_ID_2 = lambda_2 * criterion_pixel(recon_X2, X2)
        loss_KL_1_ = lambda_3 * compute_kl(mu1_)
        loss_KL_2_ = lambda_3 * compute_kl(mu2_)
        loss_cyc_1 = lambda_4 * criterion_pixel(cycle_X1, X1)
        loss_cyc_2 = lambda_4 * criterion_pixel(cycle_X2, X2)

        # Total loss
        loss_G = (
            loss_KL_1
            + loss_KL_2
            + loss_ID_1
            + loss_ID_2
            + loss_GAN_1
            + loss_GAN_2
            + loss_KL_1_
            + loss_KL_2_
            + loss_cyc_1
            + loss_cyc_2
        )
        
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        optimizer_D1.zero_grad()

        loss_D1 = criterion_GAN(D1(X1), valid) + criterion_GAN(D1(fake_X1.detach()), fake)

        loss_D1.backward()
        optimizer_D1.step()
        
        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        optimizer_D2.zero_grad()

        loss_D2 = criterion_GAN(D2(X2), valid) + criterion_GAN(D2(fake_X2.detach()), fake)

        loss_D2.backward()
        optimizer_D2.step()
        
        if i%opt.n_itr==(opt.n_itr-1):
            break
            
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()
    
    loss_G_epoch.append(loss_G.item())
    loss_D1_epoch.append(loss_D1.item())
    loss_D2_epoch.append(loss_D2.item())
    print(
        "[Epoch %d/%d] [D1 loss: %f] [D2 loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, 
           np.mean(np.array(loss_D1_epoch)), np.mean(np.array(loss_D2_epoch)),
           np.mean(np.array(loss_G_epoch)))
    )
    print('Training time for one epoch : %.1f\n' % (time.time() - t1))
    
    with torch.no_grad():
        imgs = next(iter(val_dataloader))
        E1.eval()
        E2.eval()
        G1.eval()
        G2.eval()
        
        X1 = Variable(imgs["A"].type(Tensor))
        _, Z1 = E1(X1)
        fake_X2 = G2(Z1)
        X2 = Variable(imgs["B"].type(Tensor))
        _, Z2 = E2(X2)
        fake_X1 = G1(Z2)

        for i in range(0, X1.size(0)):
            save_result(X1, i, 'real_T1', epoch)
            save_result(X2, i, 'real_T2', epoch)
            save_result(fake_X1, i, 'fake_T1', epoch)
            save_result(fake_X2, i, 'fake_T2', epoch)
        
print('Total training time : %.1f' % (time.time() - t0))
