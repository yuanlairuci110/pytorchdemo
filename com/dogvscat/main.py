from com.dogvscat.config import opt
from com.dogvscat.data.DogCatData import TrainDogCat
from torch.utils.data import DataLoader
from com.dogvscat.models import AlexNet
from com.dogvscat.models import ResNet34
import os
import torch
from torch.autograd import Variable
import tqdm

def load_data():
    train_dataset = TrainDogCat(opt.train_data_root, train=True)
    val_dataset = TrainDogCat(opt.train_data_root, train=False)
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    for ii, (data, label) in enumerate(trainloader):
        print(data.size(), label)


def train_data():
    train_dataset = TrainDogCat(opt.train_data_root, train=True)
    val_dataset = TrainDogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # 模型定义
    model = ResNet34()

    # 如果存在模型 则加载对应的参数
    if os.path.exists('./checkpoints/dogvscat_params.pkl'):
        print("存在")
        model.load_state_dict(torch.load('./checkpoints/dogvscat_params.pkl'))

    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)

    for epoch in range(opt.max_epoch):
        for ii, (data, label) in tqdm(enumerate(train_dataloader)):
            input = Variable(data)
            target = Variable(label)
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            if ii == 10:
                model.save()
        model.save()




if __name__ == '__main__':
    # 加载数据
    # load_data()

    # 训练模型
    train_data()
