import math
import random

import numpy as np

from utils import readfile


class LogisticRegression:
    def __init__(self, max_iter=2000, learning_rate=0.01, sgd=False):
        self.w = None
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.sgd = sgd

    def fit(self, x, y):
        x = np.array(np.insert(x, 0, [1], axis=1))
        self.w = np.zeros(len(x[0]))

        def sigmoid(z):
            return 1 / (1 + math.exp(-z))

        def stochastic_gradient(w):
            if self.sgd:
                i = random.randint(0, len(x) - 1)
                return sigmoid(-y[i] * np.dot(w, x[i])) * (-y[i] * x[i])
            else:
                v = 0
                for i in range(len(x)):
                    v += sigmoid(-y[i] * np.dot(w, x[i])) * (-y[i] * x[i])
                return v / len(x)

        for _ in range(self.max_iter):
            self.w -= self.learning_rate * stochastic_gradient(self.w)

    def predict(self, x):
        return np.dot(np.insert(x, 0, [1]), self.w)

    def score(self, x, y):
        err = 0
        for i in range(len(x)):
            if self.predict(x[i]) * y[i] < 0:
                err += 1
        return err / len(test[0])


if __name__ == '__main__':
    train = readfile('train.dat')
    test = readfile('test.dat')

    # gd
    reg = LogisticRegression(learning_rate=0.01, max_iter=2000)
    reg.fit(*train)
    print(reg.w)
    print(reg.score(*test))

    # sgd
    reg = LogisticRegression(learning_rate=0.01, max_iter=2000, sgd=True)
    reg.fit(*train)
    print(reg.w)
    print(reg.score(*test))
