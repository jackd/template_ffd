import numpy as np
from scipy.special import comb
from util import mesh3d


def bernstein_poly(n, v, stu):
    coeff = comb(n, v)
    weights = coeff * ((1 - stu) ** (n - v)) * (stu ** v)
    return weights


def trivariate_bernstein(stu, lattice):
    if len(lattice.shape) != 4 or lattice.shape[3] != 3:
        raise ValueError('lattice must have shape (L, M, N, 3)')
    l, m, n = (d - 1 for d in lattice.shape[:3])
    lmn = np.array([l, m, n], dtype=np.int32)
    v = mesh3d(
        np.arange(l+1, dtype=np.int32),
        np.arange(m+1, dtype=np.int32),
        np.arange(n+1, dtype=np.int32),
        dtype=np.int32)
    stu = np.reshape(stu, (-1, 1, 1, 1, 3))
    weights = bernstein_poly(n=lmn, v=v, stu=stu)
    weights = np.prod(weights, axis=-1, keepdims=True)
    return np.sum(weights * lattice, axis=(1, 2, 3))
