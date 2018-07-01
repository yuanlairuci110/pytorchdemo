import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 定义超参数

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

# 加载数据
train_data = dsets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor,
    download=DOWNLOAD_MNIST
)

# 打印一个实例
print(train_data.train_data.size())
