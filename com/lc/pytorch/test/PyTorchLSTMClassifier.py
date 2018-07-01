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

# 加载训练数据
train_data = dsets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# 打印一个实例
print(train_data.train_data.size())  # 测试数据的大小
print(train_data.train_labels.size())  # 测试数据的类别
plt.imshow(train_data.train_data[12].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[12])
plt.show()

# 加载训练数据
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 加载测试数据
test_data = dsets.MNIST(root="./mnist", train=False, transform=transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor)[:2000] / 255
test_y = test_data.test_labels.numpy().squeeze()[:2000]


# print(test_x[0])
# print(test_y)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_x) = self.rnn(x, None)

        out = self.out(r_out[:, -1, :])
        return out


# 保存LSTM循环神经网络
def save():
    rnn = RNN()

    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    # 训练和测试
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            if step == 0:
                print(b_x[0])
                print(b_y[0])
            b_x = b_x.view(-1, 28, 28)
            output = rnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = rnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
                print('Epoch: ', epoch, ' | step :', step, '| train loss: %.4f' % loss.data.numpy(),
                      '| test accuracy: %.2f' % accuracy)

    # 保存模型
    torch.save(rnn, 'rnn_lstm.pkl')
    torch.save(rnn.state_dict(), 'rnn_lstm_params.pkl')

# 通过网络模型启动
def restore_net():
    rnn2 = torch.load('rnn_lstm.pkl')
    test_output = rnn2(test_x[:100].view(-1, 28, 28))
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, '预测值')
    print(test_y[:100], '实际值')

def restore_params():
    rnn3 = RNN()
    rnn3.load_state_dict(torch.load('rnn_lstm_params.pkl'))
    test_output = rnn3(test_x[101:200].view(-1, 28, 28))
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, '预测值')
    print(test_y[101:200], '实际值')
# save()

restore_net()

restore_params()