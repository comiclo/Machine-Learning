import numpy as np

from utils import readfile


# Pocket Perceptron
class Perceptron:
    def __init__(self, max_iter=2000):
        self.w = None
        self.max_iter = max_iter
        self.iterations = 0

    def fit(self, X, Y):
        X = np.insert(X, 0, [1], axis=1)
        size, dim = X.shape
        w = np.zeros(dim)
        lowest_error = size

        def get_errors(w):
            errors = 0
            for i in range(size):
                if np.inner(w, X[i]) * Y[i] <= 0:
                    errors += 1
            return errors

        while True:
            for i in np.random.permutation(size):
                # update w
                if np.inner(w, X[i]) * Y[i] <= 0:
                    w += X[i] * Y[i]
                    self.iterations += 1

                errors = get_errors(w)
                if errors < lowest_error:
                    self.w = w.copy()
                    lowest_error = errors

                if self.iterations == self.max_iter:
                    return

    def predict(self, x):
        return -1 if np.inner(self.w, np.insert(x, 0, [1])) <= 0 else 1

    def score(self, X, Y):
        errors = 0
        for i in range(len(X)):
            if self.predict(X[i]) * Y[i] < 0:
                errors += 1
        return errors / len(X)


if __name__ == '__main__':
    train = readfile('train.dat')
    c = Perceptron()
    c.fit(*train)

    test = readfile('test.dat')
    print(c.score(*test))
    print(c.w)
