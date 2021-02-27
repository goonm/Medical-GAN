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


def save_result(img, i, tag, epoch):
    save_path = "result/epoch:%s_%s_%s.jpg" % (str(epoch+1).zfill(3), str(i).zfill(3), tag)
    img = img.cpu().detach().numpy()
    img = img[i][0]
    img = img+abs(np.min(img))
    img = img / np.max(img)*255
    cv2.imwrite(save_path, img)
    
def getLargestCC(segmentation):
    labels = label(segmentation)
    if labels.max() ==0:
        return 0
    else:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC
    
class CustomDataset(Dataset):
    def __init__(self, data_path, img_height, transforms_, mode='train'):
        self.data_path = data_path
        #self.input_shape = (int(img_height*1.12), int(img_height*1.12))
        self.input_shape = (img_height, img_height)
        self.transform = transforms.Compose(transforms_)
        self.mode = mode
        
    def __len__(self):
        return len(self.data_path)
    
    def preprocessing(self, img):
        new_img = cv2.resize(img, self.input_shape)
        new_img = (new_img-np.min(new_img)) / (np.max(new_img)-np.min(new_img))
        new_img = (new_img * 2) - 1
        return new_img
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.data_path[idx % len(self.data_path)]
        
        img = cv2.imread(img_path, 0)
        img = self.preprocessing(img)
        
        return img[None]