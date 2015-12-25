from cvxopt import matrix, solvers
import numpy as np


class HardMarginSVM:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, Y):
        d = len(X[0])
        n = len(X)
        p = matrix(np.identity(d + 1))
        p[0, 0] = 0
        q = matrix(np.zeros(d + 1))
        g = matrix(-np.diag(Y) * np.matrix(np.append(np.ones((n, 1)), X, axis=1)))
        h = matrix(-np.ones(n))
        sol = solvers.qp(p, q, g, h)['x']
        self.b = sol[0]
        self.w = np.array(matrix.trans(sol[1:, :]))

    def predict(self, x):
        return 1 if (np.inner(self.w, x) + self.b) > 0 else -1


if __name__ == '__main__':
    X = [[1, -2], [4, -5], [4, -1], [5, -2], [7, -7], [7, 1], [7, 1]]
    Y = [-1, -1, -1, 1, 1, 1, 1]
    c = HardMarginSVM()
    c.fit(X, Y)
    for i in range(len(X)):
        assert c.predict(X[i]) * Y[i] > 0
