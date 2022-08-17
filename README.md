# Differential operators on point clouds
Simple and small library to compute differential operators (gradient, divergence, Laplacian) on point clouds.

For sake of simplicity, every operation is written in Numpy and can be accelerated with Numba or Jax. If you would like to use these operators in PyTorch, please refer the github repository for [DeltaConv](https://github.com/rubenwiersma/deltaconv): `pip install deltaconv` and use the operators from `deltaconv.geometry`.

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
