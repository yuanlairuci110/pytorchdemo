# Torch 回归

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

# linspace 线形图  unsqueeze ： 把一维的数据变为二维的数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
# y = 2*x^2 + b
y = 2*x.pow(2) + 0.2*torch.rand(x.size())               # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
# torch 只能训练 Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x1, y1 = Variable(x), Variable(y)

# 画散点图 scatter 打印散点图
# plt.scatter(x1.data.numpy(), y1.data.numpy())
# print("-----------输出x和y的图形----------\n")
# plt.show()

# 建立神经网络
class Net(torch.nn.Module): # 继承 torch 的 Module
    # 定义所有层属性，搭建层所需要的信息，
    # self : 标记  n_feature : 隐藏层的输入； n_hidden :  隐藏层神经元的个数 ； n_output :
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        # 定义每层用什么样的形式
        # 隐藏层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer  隐藏层线性输出
        # 预测层
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer  输出层线性输出

    # 搭建层与层的关系链接
    # x 输入信息
    def forward(self, x):  # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))  # activation function for hidden layer  激励函数(隐藏层的线性值)  用 self.hidden 加工输入信息
        x = self.predict(x)  # linear output   输出值
        return x

# 定义神经网络
# n_feature 输入值1个 ： n_hidden ：10个隐藏层/10个神经元  n_output ：1个输出
net = Net(n_feature=1, n_hidden=10, n_output=1)
# 神经网络结构
print(net)  # net architecture

# optimizer 是训练的工具 优化神经网络
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率  学习率越高学习越快，学习率越高不很好，会忽略一些信息
# 计算误差
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss  预测值和真实值的误差计算公式 (均方差)

plt.ion()   # something about plotting 画图

for t in range(300):
    prediction = net(x)     # input x and predict based on x  给 net 训练数据 x, 输出预测值

    # 预测信息与真实值的对比，有多少不同的地方； 预测值在前，真实值在后，不然有时候两者的位数不同，结果不同；
    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)   计算两者的误差 前者是输出值，后者是目标值

    optimizer.zero_grad()   # clear gradients for next train 清空上一步的残余更新参数值  把全部参数的梯度降为0
    loss.backward()         # backpropagation, compute gradients  误差反向传播, 计算参数更新值  ；反向传递，给每个神经网络节点赋上计算的梯度值
    optimizer.step()        # apply gradients  将参数更新值施加到 net 的 parameters 上；  优化梯度

    if t % 20 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()