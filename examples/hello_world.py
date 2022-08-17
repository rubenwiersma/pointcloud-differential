import numpy as np
from pcdiff import knn_graph, estimate_basis, build_grad_div

# Random point cloud
pos = np.random.rand(1000, 3)

# Generate kNN graph
edge_index = knn_graph(pos, 20)
# Estimate normals and local frames
basis = estimate_basis(pos, edge_index)
# Build gradient and divergence operators (Scipy sparse matrices)
grad, div = build_grad_div(pos, *basis, edge_index)

# ... use gradient and divergence in any task you like