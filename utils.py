import numpy as np


def readfile(filename):
    with open(filename, 'r') as f:
        X = []
        Y = []
        for line in f:
            split = line.split()
            X.append(list(map(float, split[:-1])))
            Y.append(int(split[-1]))
        return X, Y


def readfile2(filename):
    data = np.loadtxt(filename)
    return data[:, :-1], data[:, -1]
