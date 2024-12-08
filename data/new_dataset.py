import os.path
import torch
import random
import torchvision.transforms.functional as tf
from data.base_dataset import BaseDataset, get_transform
import PIL

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path

# ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
import random

import cv2
random.seed(1)
#random.seed(2)


def mask_bboxregion(mask):
    w,h = np.shape(mask)[:2]
    valid_index = np.argwhere(mask==255) # [length,2]
    if np.shape(valid_index)[0] < 1:
        x_left = 0
        x_right = 0
        y_bottom = 0
        y_top = 0
    else:
        x_left = np.min(valid_index[:,0])
        x_right = np.max(valid_index[:,0])
        y_bottom = np.min(valid_index[:,1])
        y_top = np.max(valid_index[:,1])
    region = mask[x_left:x_right,y_bottom:y_top]
    return region
    # return [x_left, y_top, x_right, y_bottom]

def findContours(im):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4
    Returns:
        contours, hierarchy
    """
    img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)  # PILתcv2
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if cv2.__version__.startswith('4'):
    #     contours, hierarchy = cv2.findContours(*args, **kwargs)
    # elif cv2.__version__.startswith('3'):
    #     _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    # else:
    #     raise AssertionError(
    #         'cv2 must be either version 3 or 4 to call this method')
    return contours, hierarchy


class NEWDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        ## one image
        self.path_img = []
        self.path_mask = []
        
        self.opt = opt
        self._load_images_paths()
        self.transform = get_transform(opt)

    def _load_images_paths(self,):
        path = self.opt.content_dir 
        if os.path.isdir(path):
            # suo you wen jian lu jing 
            self.paths_all = list(map(str, Path(path).glob('*.jpg')))
        else:
            self.paths_all = [self.opt.content]

        for path in self.paths_all:
            # png is mask , and jpg is the pic
            path_mask = str(path).replace('.jpg','.png')
            if not os.path.exists(path_mask):
                # print('train not exist',path_mask)
                continue
            else:
                self.path_img.append(str(path))
                self.path_mask.append(path_mask)
        self.path_img.sort()
        self.path_mask.sort()

    def select_mask(self,index):
        mask_all = Image.open(self.path_mask[index]).convert('L')
        mask_array = np.array(mask_all)
        mask_value = np.unique(np.sort(mask_array[mask_array>0]))
        # 当mask全为0的时候直接返回也
        if len(mask_value != 0):
            random_pixel = random.choice(mask_value)
            if random_pixel!=255: 
                mask_array[mask_array==255] = 0
            mask_array[mask_array==random_pixel]=255
            mask_array[mask_array!=255]=0

        return mask_array

    def get_small_scale_mask(self, mask, number):
        """generate n*n patch to supervise discriminator"""
        mask = np.asarray(mask)
        mask = np.uint8(mask / 255.)
        mask_small = np.zeros([number, number],dtype=np.float32)
        split_size = self.opt.load_size // number
        for i in range(number):
            for j in range(number):
                mask_split = mask[i*split_size: (i+1)*split_size, j*split_size: (j+1)*split_size]
                mask_small[i, j] = (np.sum(mask_split) > 0) * 255
        mask_small = np.uint8(mask_small)
        return Image.fromarray(mask_small,mode='L')

    def __getitem__(self, index):
        if self.opt.is_train:
            img = Image.open(self.path_img[index]).convert('RGB')
            mask = self.select_mask(index) # 已经二值化
            mask = Image.fromarray(mask)
        else:
            img = Image.open(self.path_img[index]).convert('RGB')
            mask = Image.open(self.path_mask[index]).convert('L')

        mask_d = self.make_mask_d(mask) # PIL
        img = tf.resize(img, [self.opt.load_size, self.opt.load_size])
        mask = tf.resize(mask, [self.opt.load_size, self.opt.load_size])
        mask_d = tf.resize(mask_d, [self.opt.load_size, self.opt.load_size])
        mask_small = self.get_small_scale_mask(mask, self.opt.patch_number)

        img = self.transform(img) # [-1, 1]
        mask = tf.to_tensor(mask) # 0 or 1

        mask_d  = tf.to_tensor(mask_d)
        mask_d[mask_d >= 0.2] = 1
        mask_d[mask_d < 0.2] = 0
        
        mask_small = tf.to_tensor(mask_small) # 0 or 1
        return {'img': img, 'mask': mask, 'mask_small':mask_small, 'img_path':self.path_img[index], 'mask_d':mask_d}

    def __len__(self):
        return len(self.path_img)
    
    def make_mask_d(self, mask):
        mask_array = np.array(mask)
        kernel = np.ones((10,10), np.uint8)
        mask_d = cv2.dilate(mask_array, kernel, iterations=1)
        mask_d = Image.fromarray(mask_d)
        return mask_d # 返回的是 PIL
