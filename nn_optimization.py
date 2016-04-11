#!/usr/bin/env python

from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DATA_NUM = 200

class NeuralNetwork():
    def __init__(self, hidden):
        self.epoch = 5000
        self.alpha = 0.3
        self.hidden = hidden
        self.W1 = np.random.normal(size=(hidden, 1))
        self.b1 = np.random.normal(size=(hidden, 1))
        self.W2 = np.random.normal(size=(1, hidden))
        self.b2 = np.random.normal(size=(1, 1))
        self.minibatch_size = 50
        self.gamma = 0.9

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1.0 - (np.tanh(x) * np.tanh(x))

    def relu(self, x):
        elements = x[:, 0]
        return np.array([max(e, 0) for e in elements]).reshape(len(x), 1)

    def relu_prime(self, x):
        elements = x[:, 0]
        return np.array([1.0 if e > 0 else 0.0 for e in elements]).reshape(len(x), 1)

    def predict(self, x):
        u = np.dot(self.W1, x) + self.b1
        # z = self.sigmoid(u)
        # z = self.relu(u)
        z = self.tanh(u)
        return np.dot(self.W2, z) + self.b2

    def predict_all(self, xs):
        ret = []
        for x in xs:
            ret.append(self.predict(x)[0][0])
        return np.array(ret)

    def error(self, x, y):
        return ((self.predict(x) - y)) ** 2 / 2.0

    def gradient(self, x, y):
        # Feed forward
        u = np.dot(self.W1, x) + self.b1
        # z = self.sigmoid(u)
        # z = self.relu(u)
        z = self.tanh(u)
        v = np.dot(self.W2, z) + self.b2
        # Backpropagation
        d2 = v - y
        g2 = np.dot(d2, z.T)
        # d1 = np.dot((self.W2.T * self.sigmoid_prime(u)), d2.T)
        d1 = np.dot((self.W2.T * self.relu_prime(u)), d2.T)
        g1 = np.dot(x.T, d1)

        return (g1, d1, g2, d2)

    def train_with_sgd(self, xs, ys):
        frames = []
        error_history = []
        for i in xrange(0, self.epoch):
            e = 0.0
            for x, y in zip(xs, ys):
                (dW1, dB1, dW2, dB2) = self.gradient(x, y)
                self.W1 -= self.alpha * dW1
                self.b1 -= self.alpha * dB1
                self.W2 -= self.alpha * dW2
                self.b2 -= self.alpha * dB2
                e += self.error(x, y)[0]
            error_history.append(e)
            # frames.append(plt.plot(xs, self.predict_all(xs), 'ro'))
            if i % 100 == 0:
                print("Epoch={}, Error={}".format(i, e / len(xs)))
        return (np.array(error_history), frames)

    def train_with_momentum(self, xs, ys):
        frames = []
        error_history = []
        for i in xrange(0, self.epoch):
            g1 = np.zeros((self.hidden, 1))
            g2 = np.zeros((1, self.hidden))
            e = 0.0
            pre_dW1 = np.zeros((self.hidden, 1))
            pre_dB1 = np.zeros((self.hidden, 1))
            pre_dW2 = np.zeros((1, self.hidden))
            pre_dB2 = np.zeros((1, 1))
            for x, y in zip(xs, ys):
                (dW1, dB1, dW2, dB2) = self.gradient(x, y)
                pre_dW1 = self.gamma * pre_dW1 + self.alpha * dW1
                pre_dB1 = self.gamma * pre_dB1 + self.alpha * dB1
                pre_dW2 = self.gamma * pre_dW2 + self.alpha * dW2
                pre_dB2 = self.gamma * pre_dB2 + self.alpha * dB2
                self.W1 -= pre_dW1
                self.b1 -= pre_dB1
                self.W2 -= pre_dW2
                self.b2 -= pre_dB2
                e += self.error(x, y)[0]
            # frames.append(plt.plot(xs, self.predict_all(xs), 'ro'))
            error_history.append(e)
            if i % 100 == 0:
                print("Epoch={}, Error={}".format(i, e / len(xs)))
        return (np.array(error_history), frames)

    def train_with_nag(self, xs, ys):
        frames = []
        error_history = []
        for i in xrange(0, self.epoch):
            e = 0.0
            pre_dW1 = np.zeros((self.hidden, 1))
            pre_dB1 = np.zeros((self.hidden, 1))
            pre_dW2 = np.zeros((1, self.hidden))
            pre_dB2 = np.zeros((1, 1))
            for x, y in zip(xs, ys):
                self.W1 -= self.gamma * pre_dW1
                self.b1 -= self.gamma * pre_dB1
                self.W2 -= self.gamma * pre_dW2
                self.b2 -= self.gamma * pre_dB2
                (dW1, dB1, dW2, dB2) = self.gradient(x, y)
                self.W1 += self.gamma * pre_dW1
                self.b1 += self.gamma * pre_dB1
                self.W2 += self.gamma * pre_dW2
                self.b2 += self.gamma * pre_dB2
                pre_dW1 = self.gamma * pre_dW1 + self.alpha * dW1
                pre_dB1 = self.gamma * pre_dB1 + self.alpha * dB1
                pre_dW2 = self.gamma * pre_dW2 + self.alpha * dW2
                pre_dB2 = self.gamma * pre_dB2 + self.alpha * dB2
                self.W1 -= pre_dW1
                self.b1 -= pre_dB1
                self.W2 -= pre_dW2
                self.b2 -= pre_dB2
                e += self.error(x, y)[0]
            # frames.append(plt.plot(xs, self.predict_all(xs), 'ro'))
            error_history.append(e)
            if i % 100 == 0:
                print("Epoch={}, Error={}".format(i, e / len(xs)))
        return (np.array(error_history), frames)

    def train(self, xs, ys):
        m = len(xs)
        frames = []
        error_history = []
        for i in xrange(0, self.epoch):
            e = 0.0
            gW1 = np.zeros((self.hidden, 1))
            gB1 = np.zeros((self.hidden, 1))
            gW2 = np.zeros((1, self.hidden))
            gB2 = np.zeros((1, 1))
            for x, y in zip(xs, ys):
                (dW1, dB1, dW2, dB2) = self.gradient(x, y)
                gW1 += dW1 / m
                gB1 += dB1 / m
                gW2 += dW2 / m
                gB2 += dB2 / m
                e += self.error(x, y)[0] / m
            self.W1 -= self.alpha * gW1
            self.b1 -= self.alpha * gB1
            self.W2 -= self.alpha * gW2
            self.b2 -= self.alpha * gB2
            # frames.append(plt.plot(xs, self.predict_all(xs), 'ro'))
            error_history.append(e)
            if i % 100 == 0:
                print("Epoch={}, Error={}".format(i, e / len(xs)))
        return (np.array(error_history), frames)

    def train_with_adagrad(self, xs, ys):
        epsilon = 1.0e-8
        frames = []
        error_history = []
        for i in xrange(0, self.epoch):
            rW1 = np.zeros((self.hidden, 1))
            rB1 = np.zeros((self.hidden, 1))
            rW2 = np.zeros((1, self.hidden))
            rB2 = np.zeros((1, 1))
            e = 0.0
            for x, y in zip(xs, ys):
                (dW1, dB1, dW2, dB2) = self.gradient(x, y)
                rW1 += dW1 * dW1
                rB1 += dB1 * dB1
                rW2 += dW2 * dW2
                rB2 += dB2 * dB2
                self.W1 -= (self.alpha / (np.sqrt(rW1) + epsilon)) * dW1
                self.b1 -= (self.alpha / (np.sqrt(rB1) + epsilon)) * dB1
                self.W2 -= (self.alpha / (np.sqrt(rW2) + epsilon)) * dW2
                self.b2 -= (self.alpha / (np.sqrt(rB2) + epsilon)) * dB2
                e += self.error(x, y)[0]
            error_history.append(e)
            # frames.append(plt.plot(xs, self.predict_all(xs), 'ro'))
            if i % 100 == 0:
                print("Epoch={}, Error={}".format(i, e / len(xs)))
        return (np.array(error_history), frames)

    def train_with_rmsprop(self, xs, ys):
        epsilon = 1.0e-8
        gamma = 0.9
        frames = []
        error_history = []
        for i in xrange(0, self.epoch):
            rW1 = np.zeros((self.hidden, 1))
            rB1 = np.zeros((self.hidden, 1))
            rW2 = np.zeros((1, self.hidden))
            rB2 = np.zeros((1, 1))
            e = 0.0
            for x, y in zip(xs, ys):
                (dW1, dB1, dW2, dB2) = self.gradient(x, y)
                rW1 = gamma * rW1 + (1.0 - gamma) * dW1 * dW1
                rB1 = gamma * rB1 + (1.0 - gamma) * dB1 * dB1
                rW2 = gamma * rW2 + (1.0 - gamma) * dW2 * dW2
                rB2 = gamma * rB2 + (1.0 - gamma) * dB2 * dB2
                self.W1 -= (self.alpha / (np.sqrt(rW1) + epsilon)) * dW1
                self.b1 -= (self.alpha / (np.sqrt(rB1) + epsilon)) * dB1
                self.W2 -= (self.alpha / (np.sqrt(rW2) + epsilon)) * dW2
                self.b2 -= (self.alpha / (np.sqrt(rB2) + epsilon)) * dB2
                e += self.error(x, y)[0]
            error_history.append(e)
            # frames.append(plt.plot(xs, self.predict_all(xs), 'ro'))
            if i % 100 == 0:
                print("Epoch={}, Error={}".format(i, e / len(xs)))
        return (np.array(error_history), frames)

class LinearRegression():

    def __init__(self, degree):
        self.epoch = 10000
        self.alpha = 0.2
        self.degree = degree
        # self.W = np.random.normal(size=degree)
        self.W = np.zeros(degree)

    def predict(self, x):
        return np.dot(np.power(x, np.arange(self.degree)).T, self.W)

    def predict_all(self, xs):
        ret = []
        for x in xs:
            ret.append(self.predict(x))
        return np.array(ret)

    def error(self, x, y):
        return ((self.predict(x) - y) ** 2) / 2.0

    def gradient(self, x, y):
        return np.power(x, np.arange(self.degree)) * (self.predict(x) - y)

    def train(self, xs, ys):
        m = len(xs)
        error_history = []
        for i in xrange(0, self.epoch):
            g = np.zeros(self.degree)
            e = 0.0
            for x, y in zip(xs, ys):
                g += self.gradient(x, y) / m
                e += self.error(x, y) / m
            self.W = self.W - self.alpha * g
            error_history.append(e)
            if i % 100 == 0:
                print("Epoch={}, Error={}".format(i, e))
        return np.array(error_history)

def run_linear_regression():
    lr = LinearRegression(7)
    xs = np.linspace(0.0, 1.0, num=DATA_NUM)
    ys = np.sin(2 * np.pi * xs)
    # ys = np.cos(2 * np.pi * xs)
    noise = np.random.normal(0, 0.02, xs.size)
    history = lr.train(xs, ys + noise)
    plt.figure(1)
    plt.plot(xs, ys, 'b-', linewidth=2.0)
    plt.plot(xs, lr.predict_all(xs), 'ro')
    plt.figure(2)
    plt.plot(np.arange(len(history)), history, 'go')
    plt.show()

def run_neural_network(optimization):
    nn = NeuralNetwork(7)
    xs = np.linspace(0.0, 1.0, num=DATA_NUM)
    ys = np.sin(2 * np.pi * xs) + np.cos(5 * np.pi * xs)
    # ys = np.cos(2 * np.pi * xs)
    noise = np.random.normal(0, 0.02, xs.size)
    history = None
    print("Running NeuralNetwork with {}".format(optimization))
    if optimization == 'sgd':
        (history, frames) = nn.train_with_sgd(xs, ys + noise)
    elif optimization == 'momentum':
        (history, frames) = nn.train_with_momentum(xs, ys + noise)
    elif optimization == 'nag':
        (history, frames) = nn.train_with_nag(xs, ys + noise)
    elif optimization == 'adagrad':
        (history, frames) = nn.train_with_adagrad(xs, ys + noise)
    elif optimization == 'rmsprop':
        (history, frames) = nn.train_with_rmsprop(xs, ys + noise)
    else:
        (history, frames) = nn.train(xs, ys + noise)
    fig1 = plt.figure(1)
    # ani = animation.ArtistAnimation(fig1, frames, interval=1)
    plt.plot(xs, ys, 'b-', linewidth=2.0)
    plt.plot(xs, nn.predict_all(xs), 'r-')
    plt.figure(2)
    plt.plot(np.arange(len(history)), history, 'go')
    plt.show()

def main(options, args):
    if options.type == 'nn':
        run_neural_network(options.optimization)
    elif options.type == 'lr':
        run_linear_regression()
    else:
        print("Invalid algorithm type: {}".format(options.type))

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--type", dest="type", help="Specify algorithm type lr or nn")
    parser.add_option("-o", "--optimization", dest="optimization", help="Specify Optimization algorithm")
    (options, args) = parser.parse_args()
    main(options, args)
