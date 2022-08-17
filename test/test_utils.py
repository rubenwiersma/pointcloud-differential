import numpy as np

from pcdiff import batch_dot


def test_batch_dot():
    a = np.random.rand(1024, 10)
    b = np.random.rand(1024, 10)

    a_dot_b = (a * b).sum(axis=1, keepdims=True)
    out = batch_dot(a, b)

    assert np.allclose(out, a_dot_b)
