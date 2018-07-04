from com.dogvscat.config import opt
from com.dogvscat.data.DogCatData import DogCat
from torch.utils.data import DataLoader

train_dataset = DogCat(opt.lc_data_root)
trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

if __name__ == '__main__':
    for ii, (data, label) in enumerate(trainloader):
        print(data.size(), label)
