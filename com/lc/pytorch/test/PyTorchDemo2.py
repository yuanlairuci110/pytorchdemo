import torch
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

show = ToPILImage()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = tv.datasets.CIFAR10(root='/data', train=True, download=True, transform=transform)

tranloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = tv.datasets.CIFAR10(root='/data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

(data, label) = trainset[100]
print(classes[label])

show((data + 1) / 2).resize((100, 100))

for epoch in range(2):
    for i, data in enumerate(tranloader, 0):
        inputs, labels = data
