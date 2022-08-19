# Install polyscope to run this example:
# pip install polyscope, potpourri3d
import polyscope as ps
import potpourri3d as pp3d

import numpy.linalg as LA
from pcdiff import knn_graph, estimate_basis, build_grad_div

# Initialize polyscope
ps.init()
ps.set_ground_plane_mode('none')

# Load point cloud (vertices of mesh)
pos, _ = pp3d.read_mesh('examples/spot.obj')

# Add point cloud to Polyscope
ps.register_point_cloud("Spot", pos)
# Show x-coordinate as scalar value
ps.get_point_cloud("Spot").add_scalar_quantity("x-coordinate", pos[:, 0], enabled=True)

# Compute gradient, divergence
edge_index = knn_graph(pos, 20)
normal, x_basis, y_basis = estimate_basis(pos, edge_index)
grad, div = build_grad_div(pos, normal, x_basis, y_basis, edge_index)

# Compute gradient in tangent basis coordinates
gradient_x = grad @ pos[:, :1]

# Project into 3D
gradient_x = gradient_x.reshape(-1, 2)
gradient_x_3d = gradient_x[:, 0:1] * x_basis + gradient_x[:, 1:] * y_basis

# Add gradient vectors to Polyscope
ps.get_point_cloud("Spot").add_vector_quantity("Gradient of x-coordinate", gradient_x_3d, enabled=True)

# Compute Laplacian on points as divergence of gradient
L_pos = div @ grad @ pos

# Show result as 3D vectors on point clouds (point in the normal direction, norm is mean curvature)
ps.get_point_cloud("Spot").add_vector_quantity("Mean curvature vector", L_pos)
ps.get_point_cloud("Spot").add_scalar_quantity("Mean curvature", LA.norm(L_pos, axis=1))

# Show result
ps.show()
