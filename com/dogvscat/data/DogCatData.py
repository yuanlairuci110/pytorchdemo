import os
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T


class TrainDogCat(data.Dataset):
    def __init__(self, root, train=True):
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        imgs_num = len(imgs)
        if train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]
        self.transforms = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[5, 5, 5], std=[5, 5, 5])
        ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
