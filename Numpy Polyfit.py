import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = [0.0, 1.0, 2.0, 3.0,  4.0,  5.0]
y = [0.0, 0.8, 0.9, 0.1, -0.8, -1.0]
plt.scatter(x, y)
# plt.show()

# x = np.array(x)
# y = np.array(y)
params = np.polyfit(x, y, 1)


print("params using numpy", params)
x = [0.0, 1.0, 2.0, 3.0,  4.0,  5.0]

def func(x, m, c):
	return m*x + c

plt.plot(x, [func(x_sub, *params) for x_sub in x])



x = [0.0, 1.0, 2.0, 3.0,  4.0,  5.0]
y = [0.0, 0.8, 0.9, 0.1, -0.8, -1.0]
params, extras = curve_fit(func, x, y)

print("params using curvefit", params)

plt.show()


