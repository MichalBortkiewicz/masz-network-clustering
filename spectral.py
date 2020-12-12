import os

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import seaborn as sns
sns.set()


PLOT = False
DATA_PATH = "./data/adj_matrix"
RESULTS_PATH = "./data/results"


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
    clusters = [erdos_n_p_model(N, p_inner) for _ in range(n_clusters)]
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


def get_clusters_params(eigen_values, n_clusters):
    eigen_values = sorted(eigen_values)
    if n_clusters < 0:
        n_clusters, eigenvalues_threshold = max([(i, abs(eigen_values[i] - eigen_values[i-1]))
                                                 for i in range(1, len(eigen_values))], key=lambda x: x[1])
    else:
        eigenvalues_threshold = eigen_values[n_clusters]
    return n_clusters, eigenvalues_threshold


def load_adj_matrix(file):
    csv_path = os.path.join(DATA_PATH, file)
    if "NLK" in file:
        n_clusters = -1
    else:
        n_clusters = int(file.split(".csv")[0].split("=")[1])
    adj_matrix = pd.read_csv(csv_path, header=None).values
    return adj_matrix, n_clusters


if __name__ == "__main__":
    for file in os.listdir(DATA_PATH):
        adj_matrix, n_clusters = load_adj_matrix(file)

        L = compute_laplacian_matrix(adj_matrix)
        e, v = np.linalg.eig(L)
        n_clusters, eigenvalues_threshold = get_clusters_params(eigen_values=e, n_clusters=n_clusters)

        i = np.where(e < eigenvalues_threshold)[0]
        U = np.array(v[:, i[:]])

        km = KMeans(init='k-means++', n_clusters=n_clusters)
        km.fit(U)

        results_path = os.path.join(RESULTS_PATH, file)
        clusters = km.labels_ + 1
        df = pd.DataFrame(clusters, columns=['clusters'])
        df.index += 1
        df.to_csv(results_path, header=False)

        if PLOT:
            color = np.array(range(n_clusters))[km.labels_]
            G = nx.from_numpy_matrix(adj_matrix)
            plt.figure(figsize=(12, 12))
            nx.draw_networkx(G, node_color=color)
            plt.show()
