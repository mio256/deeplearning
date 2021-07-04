# coding: utf-8
# Your code here!

import numpy as np
import matplotlib.pylab as plt


def main():
    softmax_test(np.array([0.3, 2.0, 4.0]))


def softmax_test(a):
    y = softmax(a)
    print(a)
    print(y)
    print(np.sum(y))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def show_network():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)


def show_graph(func):
    x = np.arange(-5.0, 5.0, 0.1)
    y = func(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def init_network():
    network = {}
    network['w1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    w1, w2, w3 = network['w1'], network['w2'], network['w3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = identity_func(a3)

    return y


def identity_func(x):
    return x


def relu(x):
    return np.maximum(0, x)


def step_func(x):
    return np.array(x > 0, dtype=np.int64)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logic_pattern(callback):
    print(handler(callback, 0, 0))
    print(handler(callback, 0, 1))
    print(handler(callback, 1, 0))
    print(handler(callback, 1, 1))


def AND_natural(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    else:
        return 1


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


def handler(func, *args):
    return func(*args)


if(__name__ == "__main__"):
    main()
