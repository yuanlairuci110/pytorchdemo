from com.dogvscat.config import opt
from com.dogvscat.data.dataset import DogCat
import os
from torch.utils.data import DataLoader

train_data = DogCat(opt.train_data_root, train=True)
train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

if __name__ == '__main__':
    for ii, (data, label) in enumerate(train_dataloader):
        print(ii)
