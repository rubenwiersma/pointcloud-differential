import numpy as np

from pcdiff import norm, J, I_J, curl, laplacian, hodge_laplacian, batch_dot


def random_v(N=1024, C=16, return_components=False):
    v_norm = np.random.rand(N, C) * 5
    v_angles = np.random.rand(N, C) * 2 * np.pi
    v_x = v_norm * np.cos(v_angles)
    v_y = v_norm * np.sin(v_angles)

    v = np.stack([v_x, v_y], axis=1).reshape(-1, C)
    if return_components:
        return v, v_norm, v_angles, v_x, v_y
    return v


def test_norm():
    v, v_norm, _, _, _ = random_v(1024, 16, True)
    out = norm(v)
    assert np.allclose(out, v_norm)


def test_J():
    N = 1024
    C = 16
    v, _, _, v_x, v_y = random_v(N, C, True)

    J_v = np.stack([-v_y, v_x], axis=1).reshape(-1, C)
    out = J(v)
    assert np.allclose(out, J_v)
    dot_v_J_v = (v.reshape(-1, 2, C) * out.reshape(-1, 2, C)).sum(axis=1)
    assert np.allclose(dot_v_J_v, np.zeros_like(v_x))


def test_I_J():
    N = 1024
    C = 16
    v = random_v(N, C)
    out = I_J(v)
    assert out.shape[1] == v.shape[1] * 2
    assert np.allclose(out[:, :C], v)
    assert np.allclose(out[:, C:], J(v))

# Curl, Laplacian, and Hodge-Laplacian are tested in test_gradient