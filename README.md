# `pcdiff`: Differential operators on point clouds
Simple and small library to compute differential operators (gradient, divergence, Laplacian) on point clouds.

## Installation
The recommended installation method is by using pip:
```bash
pip install pcdiff
```

## Example usage
See `demo.py` in `examples` for a full visual demo. For a quick start:
```python
import numpy as np
from pcdiff import knn_graph, estimate_basis, build_grad_div, laplacian

# Random point cloud
pos = np.random.rand(1000, 3)

# Generate kNN graph
edge_index = knn_graph(pos, 20)
# Estimate normals and local frames
basis = estimate_basis(pos, edge_index)
# Build gradient and divergence operators (Scipy sparse matrices)
grad, div = build_grad_div(pos, *basis, edge_index)

# Setup the Laplacian as the divergence of gradient:
laplacian = -(div @ grad)
```

For sake of simplicity, every operation is written in Numpy and could be accelerated with Numba or Jax. If you would like to use these operators in PyTorch, please refer the github repository for [DeltaConv](https://github.com/rubenwiersma/deltaconv): `pip install deltaconv` and use the operators from `deltaconv.geometry`.

## Alternatives
There are many alternatives to compute discrete differential operators in Python (e.g., `potpourri3d`, `libigl`, `gptoolbox`). Most of them did not have an implementation available or exposed for gradients and divergence on point clouds. `pcdiff` is intended to be complementary to these libraries.

## Citation
If you find this library useful in your own work, please cite our paper on DeltaConv, a convolution for point clouds that uses these operators:

```bib
@Article{Wiersma2022DeltaConv,
  author    = {Ruben Wiersma, Ahmad Nasikun, Elmar Eisemann, Klaus Hildebrandt},
  journal   = {Transactions on Graphics},
  title     = {DeltaConv: Anisotropic Operators for Geometric Deep Learning on Point Clouds},
  year      = {2022},
  month     = jul,
  number    = {4},
  volume    = {41},
  doi       = {10.1145/3528223.3530166},
  publisher = {ACM},
}
```

