import numpy as np
from sklearn.neighbors import KDTree

def batch_dot(a, b):
    return np.matmul(a[:, None], b[..., None]).squeeze(-1)

def knn_graph(pos, k=20):
    return (np.repeat(np.arange(pos.shape[0]), k), KDTree(pos, leaf_size=20).query(pos, k=k)[1].flatten())