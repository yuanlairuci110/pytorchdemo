import matplotlib.pyplot as plt
import numpy as np

# 使用plt.figure定义一个图像窗口. 使用plt.plot画(x ,y)曲线. 使用plt.show显示图像.
x = np.linspace(-1,1,50)

y = x**2 +1

plt.plot(x,y)
plt.show()