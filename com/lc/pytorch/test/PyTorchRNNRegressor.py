"""
我们要用到的数据就是这样的一些数据, 我们想要用 sin 的曲线预测出 cos 的曲线.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 超参数
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

# 显示数据

# 散点分布 从0到π*2 共100个点
steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)
# sin
x_np = np.sin(steps)
# cos
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target(cos)')
plt.plot(steps, x_np, 'b-', label='input(sin)')
plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        # h_state 也要作为RNN的一个输入
        r_out, h_state = self.rnn(x, h_state)
        # 保存所有时间点的预测值
        outs = []
        # 对每一个时间点计算 output
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        # 显示outs
        # print("---------outs---------------\n",outs)
        return torch.stack(outs, dim=1), h_state


rnn = RNN()
print(rnn)
