from __future__ import division

import numpy as np
from base import Metrics


class _NumpyMetrics(Metrics):
    def sum(self, array, axis=None):
        return np.sum(array, axis=axis)

    def max(self, array, axis=None):
        return np.max(array, axis=axis)

    def min(self, array, axis=None):
        return np.min(array, axis=axis)

    def expand_dims(self, array, axis):
        return np.expand_dims(array, axis=axis)

    def sqrt(self, x):
        return np.sqrt(x)

    def top_k(self, x, k, axis=-1):
        raise NotImplementedError()

    def emd(self, p0, p1):
        # from emd import emd
        # return emd(p0, p1)

        from pyemd import emd
        n = p0.shape[0]
        dist = np.zeros((n*2, n*2))
        d0 = np.sqrt(self._dist2(p0, p1))
        dist[:n, n:] = d0
        dist[n:, :n] = d0.T

        assert(np.allclose(dist, dist.T))
        h0 = np.zeros((2*n,), dtype=np.float64)
        h0[:n] = 1.0
        h1 = np.zeros((2*n,), dtype=np.float64)
        h1[n:] = 1.0
        return emd(h0, h1, dist) / n


np_metrics = _NumpyMetrics()
