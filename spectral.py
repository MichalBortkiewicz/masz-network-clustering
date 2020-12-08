import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_circles
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
import seaborn as sns

sns.set()


def erdos_n_p_model(N, p):
    adj_matrix = np.eye(N)

    for i in range(N):
        for j in range(i, N):
            if i == j:
                adj_matrix[i, j] = 0
            else:
                adj_matrix[i, j] = np.random.choice(np.array([0, 1]), size=None, p=np.array([1 - p, p]))
                adj_matrix[j, i] = adj_matrix[i, j]
    return adj_matrix


def communities(p_inner, p_outer, N=20, n_clusters=4):
    clusters = [erdos_n_p_model(N, p_inner) for i in range(n_clusters)]

    big_adj_matrix = erdos_n_p_model(N * n_clusters, p_outer)

    for j, i in enumerate([N * cluster for cluster in range(n_clusters)]):
        big_adj_matrix[i: i + N, i: i + N] = clusters[j]

    return big_adj_matrix


def compute_degree_matrix(adj_matrix):
    D = np.diag(np.sum(np.array(adj_matrix), axis=1))
    return D


def compute_laplacian_matrix(adj_matrix):
    L = compute_degree_matrix(adj_matrix) - adj_matrix
    return L


if __name__ == "__main__":
    n_clusters = 5
    eigenvalues_smaller_than = 0.8

    adj_matrix = communities(0.8, 0.01, N=20, n_clusters=n_clusters)

    L = compute_laplacian_matrix(adj_matrix)
    e, v = np.linalg.eig(L)

    # get features for clustering (based on low eigenvalues - corresponding eigenvector contains som
    # information about the large-scale structure of the network
    i = np.where(e < eigenvalues_smaller_than)[0]
    U = np.array(v[:, i[:]])

    # cluster nodes
    km = KMeans(init='k-means++', n_clusters=n_clusters)
    km.fit(U)

    colors = ['r', 'g', 'b', 'y', 'black', 'p']
    color = np.array(colors)[km.labels_]

    G = nx.from_numpy_matrix(adj_matrix)
    plt.figure(figsize=(12, 12))
    nx.draw_networkx(G, node_color=color)
    plt.show()
