import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1,1,50)

y1 = x**2 +1
y2 = x**3+1

plt.plot(x,y1,label='y1 = x**2 +1')
plt.plot(x,y2,color = "red",linestyle='--',label='y2 = x**3 +1')
plt.legend()
plt.show()