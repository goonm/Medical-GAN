import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
import SimpleITK as sitk
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
from batchgenerators.augmentations.utils import resize_segmentation

class CustomDataset(Dataset):
    def __init__(self, data_path, img_height, transforms_, mode='train'):
        self.data_path = data_path
        #self.input_shape = (int(img_height*1.12), int(img_height*1.12))
        self.input_shape = (img_height, img_height)
        self.transform = transforms.Compose(transforms_)
        self.mode = mode
        
    def __len__(self):
        return len(self.data_path)
    
    def crop_and_pick(self, img):
        x,y,z = np.where(img!=0)
        new_img = img[min(x):max(x), min(y):max(y), min(z):max(z)]
        min_z = int(new_img.shape[0]*0.4)
        max_z = int(new_img.shape[0]*0.6)
        if self.mode=='train':
            new_z = random.randint(min_z,max_z)
        else:
            new_z = int((min_z+max_z)/2)
        new_img = new_img[new_z]
        new_img = cv2.resize(new_img, self.input_shape)
        

        new_img = (new_img-np.min(new_img)) / (np.max(new_img)-np.min(new_img))
        new_img = (new_img * 2) - 1
        return new_img
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_A_path = self.data_path[idx % len(self.data_path)]
        if self.mode=='train':
            img_B_path = self.data_path[random.randint(0, len(self.data_path)-1)]
        else:
            img_B_path = img_A_path
        
        img_A = sitk.GetArrayFromImage(sitk.ReadImage(img_A_path))[1] # T1
        img_B = sitk.GetArrayFromImage(sitk.ReadImage(img_B_path))[3] # T2
        
        img_A = self.crop_and_pick(img_A)
        img_B = self.crop_and_pick(img_B)
        
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        
        return {"A":img_A, "B":img_B}