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
