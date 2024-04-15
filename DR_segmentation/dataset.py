
import os
import random
from PIL import Image
from scipy import ndimage

import torch
from torch.utils.data import Dataset

import torchvision.transforms as T
from torchvision.transforms import functional as F

import numpy as np
from glob import glob



class RandomCrop(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)
        return image, target

def square_pad(image):
    w, h = image.size
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp)
    return F.pad(image, padding, 0, 'constant')

class OrganData(Dataset):

    def __init__(self,
                 data_path: str,
                 size: tuple = (4288, 2848),
                 transform=None):

        self.image_files = self.get_filenames(os.path.join(data_path, 'image'))
        # print(self.image_files)
        self.label_files = self.get_filenames(os.path.join(data_path, 'label_png'))
        # print(self.label_files)
        self.image_files.sort()
        self.label_files.sort()

        assert self.check_validity(self.image_files, self.label_files), \
               'inconsistent pairs of data'

        self.size = size
        self.transform = transform
        self.common_transform = T.ToTensor()



    def check_validity(self,
                       inputs: list,
                       targets: list):
        image_filenames = [i.split('/')[-1][0: 8] for i in inputs]
        label_filenames = [i.split('/')[-1][0: 8] for i in targets]
        return image_filenames == label_filenames

    def get_filenames(self,
                      path: str):
        filename_list = []
        for filename in os.listdir(path):
            filename_list.append(os.path.join(path, filename))
        return filename_list

    def __getitem__(self, idx):

        img = Image.open(self.image_files[idx])
        label = Image.open(self.label_files[idx])

        if self.transform:

            img = F.crop(img, 0, 200, 2848, 3600)
            label = F.crop(label, 0, 200, 2848, 3600)
            img = T.Resize(580)(img)
            label = T.Resize(580)(label)
            rotate_degree = random.uniform(-180, 180)
            img = F.rotate(img, rotate_degree)
            label = F.rotate(label, rotate_degree)

            Crop = RandomCrop(size=(512, 512))          
            img_crop, label_crop = Crop(img, label)
            if random.random() > 0.5: 
                img_crop_flip = F.hflip(img_crop)
                label_crop_flip = F.hflip(label_crop)
            else: 
                img_crop_flip = img_crop
                label_crop_flip = label_crop

            aug = T.Compose([
                                T.ColorJitter(brightness=0.5, contrast=0.5),
                                T.ToTensor(),
                                ])
            img = aug(img_crop_flip)
            seg_label = self.common_transform(label_crop_flip)
        else:
            img = F.crop(img, 0, 200, 2848, 3600)
            label = F.crop(label, 0, 200, 2848, 3600)
            img = square_pad(img)
            label = square_pad(label)
            img = T.Resize(512)(img)
            label = T.Resize(512)(label)
            img = self.common_transform(img)
            seg_label = self.common_transform(label)
        
        img_np = np.array(img)  # (3, 1024, 1024)
        mask_np = np.array(seg_label)   # (1, 1024, 1024)
        mask_np = np.where(mask_np > 0, 1, 0)

        sample = {"image": img_np.astype(np.float32), "label": mask_np.astype(np.float32)}
        return sample

    def __len__(self):
        return len(self.image_files)



def get_coutour_embeddings(y_true, embedding, iteration):

    y_true = y_true[0, ...]
    y_true = np.asarray(y_true)

    erosion = ndimage.binary_erosion(y_true, iterations=iteration).astype(y_true.dtype)
    contour = torch.from_numpy(y_true - erosion)
    contour_embedding = contour * embedding.cpu()
    return contour_embedding, contour

def get_background_embeddings(y_true, embedding, iteration):

    y_true = y_true[0, ...]
    y_true = np.asarray(y_true)

    dilation = ndimage.binary_dilation(y_true, iterations=iteration).astype(y_true.dtype)
    bg = torch.from_numpy(dilation - y_true)
    bg_embedding = bg * embedding.cpu()
    return bg_embedding, bg

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}