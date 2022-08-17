import numpy as np
import numpy.linalg as LA
from scipy.sparse import coo_matrix
from .utils import batch_dot

EPS = 1e-5


def estimate_basis(pos, edge_index, k=None, orientation=None):
    """Estimates a tangent basis for each point, given a k-nn graph and positions.

    Args:
        pos (Tensor): an [N, 3] tensor with the point positions.
        edge_index (Tensor): indices of the adjacency matrix of the k-nn graph [2, N * k].
        k (int, optional): the number of neighbors per point,
            is derived from edge_index when no k is provided (default: None).
        orientation (Tensor, optional): an [N, 3] tensor with a rough direction of the normal to
            orient the estimated normals.
    """
    row, col = edge_index
    k = (row == 0).sum() if k is None else k
    row, col = row.reshape(-1, k), col.reshape(-1, k)
    local_pos = (pos[col] - pos[row]).transpose(0, 2, 1)
    
    # SVD to estimate bases
    U, _, _ = LA.svd(local_pos)
    
    # Normal corresponds to smallest singular vector and normalize
    normal = U[:, :, 2]
    normal = normal / LA.norm(normal, axis=-1, keepdims=True).clip(EPS)

    # If normals are given, orient using the given normals
    if orientation is not None:
        normal = np.where(batch_dot(normal, orientation) < 0, -normal, normal)

    # X axis to largest singular vector and normalize
    x_basis = U[:, :, 0]
    x_basis = x_basis / LA.norm(x_basis, axis=-1, keepdims=True).clip(EPS)
    
    # Create orthonormal basis by taking cross product
    y_basis = np.cross(normal, x_basis)
    y_basis = y_basis / LA.norm(y_basis, axis=-1, keepdims=True).clip(EPS)
    
    return normal, x_basis, y_basis


def build_tangent_basis(normal):
    """Constructs an orthonormal tangent basis, given a normal vector.

    Args:
        normal (Tensor): an [N, 3] tensor with normals per point.
    """

    # Pick an arbitrary basis vector that does not align too much with the normal
    testvec = np.tile(np.array([[1, 0, 0]]), (normal.shape[0], 1))
    testvec_alt = np.tile(np.array([[0, 1, 0]]), (normal.shape[0], 1))
    testvec = np.where(np.abs(batch_dot(normal, testvec)) > 0.9, testvec_alt, testvec)

    # Derive x basis using cross product and normalize
    x_basis = np.cross(testvec, normal)
    x_basis = x_basis / LA.norm(x_basis, axis=-1, keepdims=True).clip(EPS)

    # Derive y basis using cross product and normalize
    y_basis = np.cross(normal, x_basis)
    y_basis = y_basis / LA.norm(y_basis, axis=-1, keepdims=True).clip(EPS)
    return x_basis, y_basis


def coords_projected(pos, normal, x_basis, y_basis, edge_index, k=None):
    """Projects neighboring points to the tangent basis
    and returns the local coordinates.

    Args:
        pos (Tensor): an [N, 3] tensor with the point positions.
        normal (Tensor): an [N, 3] tensor with normals per point.
        x_basis (Tensor): an [N, 3] tensor with x basis per point.
        y_basis (Tensor): an [N, 3] tensor with y basis per point.
        edge_index (Tensor): indices of the adjacency matrix of the k-nn graph [2, N * k].
        k (int): the number of neighbors per point.
    """
    row, col = edge_index
    k = (row == 0).sum() if k is None else k

    # Compute coords
    normal = np.tile(normal[:, None], (1, k, 1)).reshape(-1, 3)
    x_basis = np.tile(x_basis[:, None], (1, k, 1)).reshape(-1, 3)
    y_basis = np.tile(y_basis[:, None], (1, k, 1)).reshape(-1, 3)
    local_pos = pos[col] - pos[row]
    local_pos = local_pos - normal * batch_dot(local_pos, normal)
    x_pos = batch_dot(local_pos, x_basis).flatten()
    y_pos = batch_dot(local_pos, y_basis).flatten()
    coords = np.stack([x_pos, y_pos], axis=1)

    return coords


def gaussian_weights(dist, k, kernel_width=1):
    """Computes gaussian weights per edge and normalizes the sum per neighborhood.

    Args:
        dist (Tensor): an [N * k] tensor with the geodesic distance of each edge.
        k (int): the number of neighbors per point.
        kernel_width (float, optional): the size of the kernel,
            relative to the average edge length in each shape (default: 1).
    """
    dist = dist.reshape(-1, k)
    avg_dist = dist.mean(axis=1, keepdims=True)
    weights = np.exp(- dist ** 2 / (kernel_width * avg_dist) ** 2)
    weights = weights / weights.sum(axis=1, keepdims=True).clip(EPS)
    
    return weights.flatten()


def weighted_least_squares(coords, weights, k, regularizer, shape_regularizer=None):
    """Solves a weighted least squares equation (see http://www.nealen.net/projects/mls/asapmls.pdf).
    In practice, we compute the inverse of the left-hand side of a weighted-least squares problem:
        B^TB c = B^Tf(x).

    This inverse can be multiplied with the right hand side to find the coefficients
    of a second order polynomial that approximates f(x).
        c = (BTB)^-1 B^T f(x).
    
    The weighted least squares problem is regularized by adding a small value \lambda
    to the diagonals of the matrix on the left hand side of the equation:
        B^TB + \lambda I.
    """
    # Setup polynomial basis
    coords_const = np.concatenate([np.ones((coords.shape[0], 1)), coords], axis=1)
    B = np.matmul(np.expand_dims(coords_const, -1), np.expand_dims(coords_const, -2))
    triu = np.triu_indices(3)
    B = B[:, triu[0], triu[1]]
    B = B.reshape(-1, k, 6) # [1, x, y, x**2, xy, y**2]

    # Compute weighted least squares
    lI = regularizer * np.eye(6, 6)[None]
    BT = (weights.reshape(-1, k, 1) * B).transpose(0, 2, 1)
    BTB = BT @ B + lI
    BTB_inv = LA.inv(BTB)
    wls = (BTB_inv @ BT).transpose(0, 2, 1).reshape(-1, 6)

    if shape_regularizer is not None:
        lI = shape_regularizer * np.eye(6, 6)[None]
        BTB = BT @ B + lI
        BTB_inv = LA.inv(BTB)
        wls_shape = (BTB_inv @ BT).transpose(0, 2, 1).reshape(-1, 6)
        return wls, wls_shape
    return wls


def fit_vector_mapping(pos, normal, x_basis, y_basis, edge_index, wls, coords):
    """Finds the transformation between a basis at point pj
    and the basis at point pi pushed forward to pj.

    See equation (15) in the supplement of DeltaConv for more details.
    """
    row, col = edge_index
    k = (row == 0).sum()

    # Compute the height over the patch by projecting the relative positions onto the normal
    patch_f = batch_dot(normal[row], pos[col] - pos[row])
    coefficients = np.sum((wls * patch_f).reshape(-1, k, 6), axis=1)
    if coefficients.shape[0] < row.max():
        coefficients = np.repeat(coefficients, k, axis=0)

    # Equation (3) and (4) from supplement
    h_x = (coefficients[row, 1] + 2 * coefficients[row, 3] * coords[:, 0] + coefficients[row, 4] * coords[:, 1])
    h_y = (coefficients[row, 2] + coefficients[row, 4] * coords[:, 0] + 2 * coefficients[row, 5] * coords[:, 1])

    # Push forward bases to p_j
    # In equation (15): \Gamma(u_j, v_j)
    gamma_x = x_basis[row] + normal[row] * h_x[..., None]
    gamma_y = y_basis[row] + normal[row] * h_y[..., None]

    # Determine inverse metric for mapping
    # Inverse metric is given in equation (9) of supplement
    det_metric = (1 + h_x ** 2 + h_y ** 2)
    E, F, G = 1 + h_x ** 2, h_x * h_y, 1 + h_y ** 2
    inverse_metric = np.stack([
        G, -F,
        -F, E
    ], axis=-1).reshape(-1, 2, 2)
    inverse_metric = inverse_metric / det_metric.reshape(-1, 1, 1)
    basis_transformation = np.concatenate([
        batch_dot(gamma_x, x_basis[col]),
        batch_dot(gamma_x, y_basis[col]),
        batch_dot(gamma_y, x_basis[col]),
        batch_dot(gamma_y, y_basis[col])
    ], axis=1).reshape(-1, 2, 2)
    
    # Compute mapping of vectors
    return inverse_metric @ basis_transformation # [N, 2, 2]


def build_grad_div(pos, normal, x_basis, y_basis, edge_index, kernel_width=1, regularizer=1e-8, shape_regularizer=None):
    """Builds a gradient and divergence operators using Weighted Least Squares (WLS).
    Note: this function is only faster if used on the GPU.
    Use pointcloud-ops when applying transforms on the CPU.

    Args:
        pos (Tensor): an [N, 3] tensor with the point positions.
        normal (Tensor): an [N, 3] tensor with normals per point.
        x_basis (Tensor): an [N, 3] tensor with x basis per point.
        y_basis (Tensor): an [N, 3] tensor with y basis per point.
        edge_index (Tensor): indices of the adjacency matrix of the k-nn graph [2, N * k].
        batch (Tensor): an [N] tensor denoting which batch each shape belongs to (default: None).
        kernel_width (float, optional): the size of the kernel,
            relative to the average edge length in each shape (default: 1).
        regularizer (float: optional): the regularizer parameter
            for weighted least squares fitting (default: 1e-8).
        normalized (bool: optional): Normalizes the operators by the
            infinity norm if set to True (default: True):
            G = G / |G|_{\inf}
        shape_regularizer (float: optional): sets the regularizer parameter
            for weighted least squares fitting of the surface, rather than the signal on the surface.
            By default, this is set to None and the same value is used for the surface and the signal.
    """

    row, col = edge_index
    k = (row == 0).sum()

    # Get coordinates in tangent plane by projecting along the normal of the plane
    coords = coords_projected(pos, normal, x_basis, y_basis, edge_index, k)

    # Compute weights based on distance in euclidean space
    dist = LA.norm(pos[col] - pos[row], axis=1)
    weights = gaussian_weights(dist, k, kernel_width)

    # Get weighted least squares result
    # wls multiplied with a function f at k neighbors will give the coefficients c0-c5
    # for the surface f(x, y) = [x, y, c0 + c1*x + c2*y + c3*x**2 + c4*xy + c5*y**2]
    # defined on a neighborhood of each point.
    if shape_regularizer is None:
        wls = weighted_least_squares(coords, weights, k, regularizer)
    else:
        wls, wls_shape = weighted_least_squares(coords, weights, k, regularizer, shape_regularizer)

    # Format as sparse matrix

    # The gradient of f at (0, 0) will be
    # df/dx|(0, 0) = [1, 0, c1 + 2*c3*0 + c4*0] = [1, 0, c1]
    # df/dy|(0, 0) = [0, 1, c2 + c4*0 + 2*c5*0] = [0, 1, c2]
    # Hence, we can use the row in wls that outputs c1 and c2 for the gradient
    # in x direction and y direction, respectively
    grad_row = np.stack([row * 2, row * 2 + 1], axis=1).flatten()
    grad_col = np.stack([col]*2, axis=1).flatten()
    grad_values = np.stack([wls[:, 1], wls[:, 2]], axis=1).flatten()

    # Create gradient matrix
    grad = coo_matrix((grad_values, (grad_row, grad_col)), shape=(pos.shape[0] * 2, pos.shape[0]))

    # Divergence
    if shape_regularizer is not None:
        wls = wls_shape
    vector_mapping = fit_vector_mapping(pos, normal, x_basis, y_basis, (row, col), wls, coords)

    # Store as sparse tensor 
    grad_vec = grad_values.reshape(-1, 1, 2)
    div_vec = (grad_vec @ vector_mapping).flatten()
    div_row = np.stack([row] * 2, axis=1).flatten()
    div_col = np.stack([col * 2, col * 2 + 1], axis=1).flatten()
    div = coo_matrix((div_vec, (div_row, div_col)), shape=(pos.shape[0], pos.shape[0] * 2))

    return grad, div
