import numpy as np
from numpy.linalg import inv

from utils import readfile


class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, x, y):
        X = np.matrix(np.insert(x, 0, [1], axis=1))
        Y = np.matrix(y)
        W = inv(X.transpose() * X) * X.transpose() * Y.transpose()
        self.w = np.asarray(W).reshape(-1)

    def predict(self, x):
        return -1 if np.dot(np.insert(x, 0, [1]), self.w) < 0 else 1


if __name__ == '__main__':
    train = readfile('train.dat')
    reg = LinearRegression()
    reg.fit(*train)

    test = readfile('test.dat')
    errors = 0
    for i in range(len(test[0])):
        if reg.predict(test[0][i]) * test[1][i] < 0:
            errors += 1
    print(errors / len(test[0]))
