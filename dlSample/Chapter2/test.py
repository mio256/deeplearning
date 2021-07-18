import matplotlib.pyplot as plt
import numpy as np


def softmax_function(x):
    return np.exp(x) / np.sum(np.exp(x))  # ソフトマックス関数


x = np.linspace(0, 600)
y = softmax_function(x)

plt.plot(x, y)
plt.show()
