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
from torchvision.utils import save_image

from models import *
from datasets import *

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from batchgenerators.utilities.file_and_folder_operations import *


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=1000, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")

opt = parser.parse_args()
print(opt)

# Create result directories
os.makedirs("result", exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

device = torch.device('cuda:0')
print('torch.cuda: ',torch.cuda.is_available())

input_shape = (opt.channels, opt.img_size, opt.img_size)

# Loss function
adversarial_loss = torch.nn.MSELoss()

# Initialize models
coupled_generators = CoupledGenerators(opt.img_size, opt.latent_dim, opt.channels)
coupled_discriminators = CoupledDiscriminators(opt.img_size, opt.channels)

# to cuda
coupled_generators.to(device)
coupled_discriminators.to(device)
adversarial_loss.to(device)

# Initialize weights
coupled_generators.apply(weights_init_normal)
coupled_discriminators.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(coupled_generators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(coupled_discriminators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor

# Image transformations
transforms_ = [
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
]

# Configure data loader
data_path = subfiles('data/train', suffix='.nii.gz')
train_dataset = CustomDataset(data_path, opt.img_size, transforms_, mode='train')
dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)


# ----------
#  Training
# ----------
t0 = time.time()
for epoch in range(opt.n_epochs):
    t1 = time.time()
    loss_G_epoch=[]
    loss_D_epoch=[]
    
    for i, batch in enumerate(dataloader):
        
        # Set model input
        imgs1 = Variable(batch["A"].type(Tensor))
        imgs2 = Variable(batch["B"].type(Tensor))
        
        batch_size = imgs1.size(0)
        
        # Adversarial ground truths
        valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        gen_imgs1, gen_imgs2 = coupled_generators(z)
        # Determine validity of generated images
        validity1, validity2 = coupled_discriminators(gen_imgs1, gen_imgs2)

        g_loss = (adversarial_loss(validity1, valid) + adversarial_loss(validity2, valid)) / 2
        
        g_loss.backward()
        optimizer_G.step()
        
        # ----------------------
        #  Train Discriminators
        # ----------------------

        optimizer_D.zero_grad()

        # Determine validity of real and generated images
        validity1_real, validity2_real = coupled_discriminators(imgs1, imgs2)
        validity1_fake, validity2_fake = coupled_discriminators(gen_imgs1.detach(), gen_imgs2.detach())

        d_loss = (
            adversarial_loss(validity1_real, valid)
            + adversarial_loss(validity1_fake, fake)
            + adversarial_loss(validity2_real, valid)
            + adversarial_loss(validity2_fake, fake)
        ) / 4
        
        d_loss.backward()
        optimizer_D.step()
        
        loss_G_epoch.append(g_loss.item())
        loss_D_epoch.append(d_loss.item())
        

        
    print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, np.mean(np.array(loss_D_epoch)), np.mean(np.array(loss_G_epoch)))
    )
    print('Training time for one epoch : %.1f\n' % (time.time() - t1))
    
    
    save_path = "result/epoch:%s_T1.jpg" % (str(epoch+1).zfill(3))
    save_data = gen_imgs1[0][0].cpu().detach().numpy()
    save_data = (save_data+abs(np.min(save_data)))
    save_data = save_data / np.max(save_data) *255
    cv2.imwrite(save_path, save_data)

    save_path = "result/epoch:%s_T2.jpg" % (str(epoch+1).zfill(3))
    save_data = gen_imgs2[0][0].cpu().detach().numpy()
    save_data = (save_data+abs(np.min(save_data)))
    save_data = save_data / np.max(save_data) *255
    cv2.imwrite(save_path, save_data)
        
print('Total training time : %.1f' % (time.time() - t0))