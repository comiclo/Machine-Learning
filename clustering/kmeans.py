import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# http://cs229.stanford.edu/notes/cs229-notes7a.pdf

class KMeans:
    def __init__(self, k, max_iter=300):
        self.k = k
        self.max_iter = max_iter
        self.centers = []
        self.clusters = []

    def fit(self, X):
        if not self.centers:
            self.centers = [np.array(X[i]) for i in random.sample(range(len(X)), self.k)]

        it = 0
        while it < self.max_iter:
            self.clusters = [[] for _ in range(self.k)]
            for x in X:
                distances = []
                for center in self.centers:
                    distances.append(np.linalg.norm(center - x))
                self.clusters[np.argmin(distances)].append(x)
            self.centers = []
            for cluster in self.clusters:
                if cluster:
                    self.centers.append(np.mean(cluster, axis=0))
            it += 1

    def plot(self):
        colors = ['r', 'b', 'g']
        if len(self.centers[0]) == 2:
            if any(self.clusters):
                for cluster in self.clusters:
                    plt.plot(*np.array(cluster).transpose(), marker='o', color=colors.pop(), ls='')
            plt.show()
        elif len(self.centers[0]) == 3:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            if any(self.clusters):
                for cluster in self.clusters:
                    ax.plot(*np.array(cluster).transpose(), marker='o', color=colors.pop(), ls='')
            plt.show()


if __name__ == '__main__':
    iris = load_iris()
    x = iris.data

    pca = PCA(n_components=3)
    x = pca.fit_transform(x)

    k = 3
    c = KMeans(k)
    c.fit(x)
    c.plot()
