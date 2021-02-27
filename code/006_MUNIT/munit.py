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
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--n_itr", type=int, default=600, help="number of iterations")
parser.add_argument("--batch_size", type=int, default=6, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=4)
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--style_dim", type=int, default=8)
opt = parser.parse_args()
print(opt)

# Create result directories
os.makedirs("result", exist_ok=True)

# Losses
criterion_recon = torch.nn.L1Loss()

device = torch.device('cuda:0')
print('torch.cuda: ',torch.cuda.is_available())

# Initialize encoders, generators and discriminators
Enc1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec1 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Enc2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec2 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
D1 = MultiDiscriminator()
D2 = MultiDiscriminator()

# to cuda
Enc1.to(device)
Dec1.to(device)
Enc2.to(device)
Dec2.to(device)
D1.to(device)
D2.to(device)
criterion_recon.to(device)

# Initialize weights
Enc1.apply(weights_init_normal)
Dec1.apply(weights_init_normal)
Enc2.apply(weights_init_normal)
Dec2.apply(weights_init_normal)
D1.apply(weights_init_normal)
D2.apply(weights_init_normal)

# Loss weights
lambda_gan = 1
lambda_id = 10
lambda_style = 1
lambda_cont = 1
lambda_cyc = 0

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),
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
data_path1 = subfiles('data/train/normal', suffix='.jpg')
data_path2 = subfiles('data/train/lung_opacity', suffix='.jpg')
data_path1_val = subfiles('data/val/normal', suffix='.jpg')
data_path2_val = subfiles('data/val/lung_opacity', suffix='.jpg')

train_dataset1 = CustomDataset(data_path1, opt.img_height, transforms_, mode='train')
val_dataset1 = CustomDataset(data_path1_val, opt.img_height, val_transforms_, mode='test')
train_dataset2 = CustomDataset(data_path2, opt.img_height, transforms_, mode='train')
val_dataset2 = CustomDataset(data_path2_val, opt.img_height, val_transforms_, mode='test')

dataloader1 = DataLoader(train_dataset1, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
val_dataloader1 = DataLoader(val_dataset1, batch_size=4, shuffle=False, num_workers=1)
dataloader2 = DataLoader(train_dataset2, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
val_dataloader2 = DataLoader(val_dataset2, batch_size=4, shuffle=False, num_workers=1)

# ----------
#  Training
# ----------
t0 = time.time()

# Adversarial ground truths
valid = 1
fake = 0

for epoch in range(opt.epoch, opt.n_epochs):
    t1 = time.time()
    loss_G_epoch=[]
    loss_D1_epoch=[]
    loss_D2_epoch=[]
    
    for i, (batch1, batch2) in enumerate(zip(dataloader1, dataloader2)):
        # Set model input
        X1 = Variable(batch1.type(Tensor))
        X2 = Variable(batch2.type(Tensor))
        
        # Sampled style codes
        style_1 = Variable(torch.randn(X1.size(0), opt.style_dim, 1, 1).type(Tensor))
        style_2 = Variable(torch.randn(X1.size(0), opt.style_dim, 1, 1).type(Tensor))
        
        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------
        Enc1.train()
        Dec1.train()
        Enc2.train()
        Dec2.train()
        D1.train()
        D2.train()
        
        optimizer_G.zero_grad()

        # Get shared latent representation
        c_code_1, s_code_1 = Enc1(X1)
        c_code_2, s_code_2 = Enc2(X2)

        # Reconstruct images
        X11 = Dec1(c_code_1, s_code_1)
        X22 = Dec2(c_code_2, s_code_2)

        # Translate images
        X21 = Dec1(c_code_2, style_1)
        X12 = Dec2(c_code_1, style_2)

        # Cycle translation
        c_code_21, s_code_21 = Enc1(X21)
        c_code_12, s_code_12 = Enc2(X12)
        X121 = Dec1(c_code_12, s_code_1) if lambda_cyc > 0 else 0
        X212 = Dec2(c_code_21, s_code_2) if lambda_cyc > 0 else 0

        # Losses
        loss_GAN_1 = lambda_gan * D1.compute_loss(X21, valid)
        loss_GAN_2 = lambda_gan * D2.compute_loss(X12, valid)
        loss_ID_1 = lambda_id * criterion_recon(X11, X1)
        loss_ID_2 = lambda_id * criterion_recon(X22, X2)
        loss_s_1 = lambda_style * criterion_recon(s_code_21, style_1)
        loss_s_2 = lambda_style * criterion_recon(s_code_12, style_2)
        loss_c_1 = lambda_cont * criterion_recon(c_code_12, c_code_1.detach())
        loss_c_2 = lambda_cont * criterion_recon(c_code_21, c_code_2.detach())
        loss_cyc_1 = lambda_cyc * criterion_recon(X121, X1) if lambda_cyc > 0 else 0
        loss_cyc_2 = lambda_cyc * criterion_recon(X212, X2) if lambda_cyc > 0 else 0

        # Total loss
        loss_G = (
            loss_GAN_1
            + loss_GAN_2
            + loss_ID_1
            + loss_ID_2
            + loss_s_1
            + loss_s_2
            + loss_c_1
            + loss_c_2
            + loss_cyc_1
            + loss_cyc_2
        )
        
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        optimizer_D1.zero_grad()

        loss_D1 = D1.compute_loss(X1, valid) + D1.compute_loss(X21.detach(), fake)

        loss_D1.backward()
        optimizer_D1.step()
        
        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        optimizer_D2.zero_grad()

        loss_D2 = D2.compute_loss(X2, valid) + D2.compute_loss(X12.detach(), fake)

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
        imgs1 = next(iter(val_dataloader1))
        imgs2 = next(iter(val_dataloader2))
        Enc1.eval()
        Dec1.eval()
        Enc2.eval()
        Dec2.eval()
        
        X1 = Variable(imgs1.type(Tensor))
        X2 = Variable(imgs2.type(Tensor))
        
        for i in range(1,5):
            # Sampled style codes
            style_1 = Variable(torch.randn(X1.size(0), opt.style_dim, 1, 1).type(Tensor))
            style_2 = Variable(torch.randn(X1.size(0), opt.style_dim, 1, 1).type(Tensor))

            c_code_1, s_code_1 = Enc1(X1)
            c_code_2, s_code_2 = Enc2(X2)

            # Translate images
            X21 = Dec1(c_code_2, style_1)
            X12 = Dec2(c_code_1, style_2)

            save_result(X12, 0, 'fake_lung_opacity_'+str(i), epoch)
            save_result(X21, 0, 'fake_normal_'+str(i), epoch)
        
print('Total training time : %.1f' % (time.time() - t0))
