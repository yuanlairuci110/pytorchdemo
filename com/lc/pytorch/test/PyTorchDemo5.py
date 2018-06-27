import torch as t
from torch.utils import data
import os
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

transform = T.Compose([T.Scale(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[5, 5, 5], std=[5, 5, 5])])


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        # 所有图片的绝对路径
        # 这里不实际加载图片，只是指定路径，当调用__getitem__时才会真正读图片
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # dog->1， cat->0
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


dataset = DogCat('./dogvscat/dogandcat/', transforms=transform)

img, label = dataset[0]
for img, label in dataset:
    print(img.size(), img.float().mean(), " dog->1， cat->0 label :", label)

dataset = ImageFolder('./dogvscat/dogcat_2/')

print(dataset.class_to_idx)

print(dataset.imgs)

print(dataset[0][1])

print(dataset[0][0])

