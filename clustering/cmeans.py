from time import time

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# ref http://home.deib.polimi.it/matteucc/Clustering/tutorial_html/cmeans.html


def cmeans(x, n_centers, degree=2, max_iter=100, tolerance=1e-2):
    n = len(x)
    u = np.zeros((n, n_centers))
    # cluster centers
    c = np.zeros((n_centers, len(x[0])))

    # initial U
    for i in range(n):
        l = np.random.sample(n_centers)
        u[i] = l / sum(l)

    error = float('Inf')

    while error > tolerance:
        # update c
        for j in range(n_centers):
            up = 0
            lo = 0
            for i in range(n):
                u_m = u[i][j] ** degree
                up += u_m * x[i]
                lo += u_m
            c[j] = up / lo

        old_U = u.copy()

        # update U
        for i in range(n):
            for j in range(n_centers):
                s = 0
                for k in range(n_centers):
                    up = norm(x[i] - c[j])
                    lo = norm(x[i] - c[k])
                    s += (up / lo) ** (2 / (degree - 1))
                u[i][j] = 1 / s

        # compute error
        error = norm(old_U - u)

    return u


def exmple_2d():
    iris = load_iris()
    x = iris.data
    y = iris.target

    pca = PCA(n_components=2)
    x = pca.fit_transform(x)

    u = cmeans(x, 3)

    # plot
    for i in range(len(x)):
        plt.plot(x[i][0], x[i][1], 'o', color=[u[i][j] for j in range(3)])
    plt.show()


def example_3d():
    iris = load_iris()
    x = iris.data
    y = iris.target

    pca = PCA(n_components=3)
    x = pca.fit_transform(x)

    u = cmeans(x, 3)

    # plot 3D
    n = len(x)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(n):
        ax.scatter(x[i][0], x[i][1], x[i][2], marker='o', color=[u[i][j] for j in range(3)])
    plt.show()


if __name__ == '__main__':
    # exmple_2d()
    example_3d()
