import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ref http://home.deib.polimi.it/matteucc/Clustering/tutorial_html/cmeans.html

iris = load_iris()
x = iris.data
y = iris.target

pca = PCA(n_components=3)
x = pca.fit_transform(x)

# degree
m = 2
# data length
N = len(x)
# cluster number
C = 3

U = np.zeros((N, C))
# cluster centers
c = np.zeros((C, len(x[0])))


# initial U
for i in range(N):
    l = np.random.sample(C)
    U[i] = l / sum(l)

max_iter = 300
tolerance = 0.01
error = 100

while error > tolerance:
    # update c
    for j in range(C):
        up = 0
        lo = 0
        for i in range(N):
            u_m = U[i][j] ** m
            up += u_m * x[i]
            lo += u_m
        c[j] = up / lo

    old_U = U.copy()

    # update U
    for i in range(N):
        for j in range(C):
            s = 0
            for k in range(C):
                up = np.linalg.norm(x[i] - c[j])
                lo = np.linalg.norm(x[i] - c[k])
                s += (up / lo) ** (2 / (m - 1))
            U[i][j] = 1 / s

    # compute error
    error = np.linalg.norm(old_U - U)

# plot
# for i in range(N):
#     plt.plot(x[i][0], x[i][1], 'o', color=[U[i][j] for j in range(3)])
# plt.show()

# plot 3D
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(N):
    ax.scatter(x[i][0], x[i][1], x[i][2], marker='o', color=[U[i][j] for j in range(3)])
plt.show()
