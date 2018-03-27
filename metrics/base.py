class Metrics(object):
    def sum(self, array, axis=None):
        raise NotImplementedError('Abstract method')

    def max(self, array, axis=None):
        raise NotImplementedError('Abstract method')

    def min(self, array, axis=None):
        raise NotImplementedError('Abstract method')

    def expand_dims(self, array, axis):
        raise NotImplementedError('Abstract method')

    def sqrt(self, x):
        raise NotImplementedError('Abstract method')

    def top_k(self, x, axis):
        raise NotImplementedError('Abstract method')

    def _size_check(self, s1, s2):
        for s1s, s2s in zip(s1.shape[:-2], s2.shape[:-2]):
            if s1s != s2s and not (s1s == 1 or s2s == 1):
                raise ValueError(
                    'Invalid shape for s1, s2: %s, %s'
                    % (str(s1.shape), str(s2.shape)))
        if s1.shape[-1] != s2.shape[-1]:
            raise ValueError(
                'last dim of s1 and s2 must be same, but got %d, %d'
                % (s1.shape[-1], s2.shape[-1]))

    def _dist2(self, s1, s2):
        s1 = self.expand_dims(s1, axis=-2)
        s2 = self.expand_dims(s2, axis=-3)
        # diff = s1 - s2
        # return self.sum(diff*diff, axis=-1)
        return self.sum((s1 - s2)**2, axis=-1)

    def _unidirectional_chamfer(self, dist2, reverse=False):
        return self.sum(self.min(dist2, axis=-2 if reverse else -1), axis=-1)

    def unidirectional_chamfer(self, s1, s2, reverse=False):
        self._size_check(s1, s2)
        dist2 = self._dist2(s1, s2)
        return self._unidirectional_chamfer(dist2, reverse=reverse)

    def _bidirectional_chamfer(self, s1, s2):
        dist2 = self._dist2(s1, s2)
        return self._unidirectional_chamfer(dist2, reverse=False) + \
            self._unidirectional_chamfer(dist2, reverse=True)

    def bidirectional_chamfer(self, s1, s2):
        self._size_check(s1, s2)
        return self._bidirectional_chamfer(s1, s2)

    def chamfer(self, s1, s2):
        return self.bidirectional_chamfer(s1, s2)

    def _unidirectional_n_chamfer(self, n, neg_dist2, reverse=False):
        values, _ = self.top_k(neg_dist2, n, axis=-2 if reverse else -1)
        return -self.sum(values, axis=(-2, -1))

    def unidirectional_n_chamfer(self, n, s1, s2, reverse=False):
        self._check_sizes(s1, s2)
        neg_dist2 = -self.dist2(s1, s2)
        return self._unidirectional_n_chamfer(n, neg_dist2, reverse=reverse)

    def _bidirectional_n_chamfer(self, n, s1, s2):
        neg_dist2 = -self._dist2(s1, s2)
        return self._unidirectional_n_chamfer(n, neg_dist2, reverse=False) + \
            self._unidirectional_n_chamfer(n, neg_dist2, reverse=True)

    def bidirectional_n_chamfer(self, n, s1, s2):
        self._size_check(s1, s2)
        return self._bidirectional_n_chamfer(n, s1, s2)

    def n_chamfer(self, n, s1, s2):
        return self.bidirectional_n_chamfer(n, s1, s2)

    def _unidirectional_hausdorff(self, dist2, reverse=False):
        return self.max(self.min(dist2, axis=-2 if reverse else -1), axis=-1)

    def unidirectional_hausdorff(self, s1, s2, reverse=False):
        self._size_check(s1, s2)
        return self._unidirectional_hausdorff(s1, s2, reverse=reverse)

    def _bidirectional_hausdorff(self, s1, s2):
        dist2 = self._dist2(s1, s2)
        return max(
            self._unidirectional_hausdorff(dist2, reverse=False),
            self._unidirectional_hausdorff(dist2, reverse=True))

    def bidirectional_hausdorff(self, s1, s2):
        self._size_check(s1, s2)
        return self._bidirectional_hausdorff(s1, s2)

    def hausdorff(self, s1, s2):
        return self.bidirectional_hausdorff(s1, s2)

    def unidirectional_modified_chamfer(self, s1, s2, reverse=False):
        self._size_check(s1, s2)
        dist2 = self._dist2(s1, s2)
        dist = self.sqrt(dist2)
        return self._unidirectional_chamfer(dist, reverse=reverse)

    def _bidirectional_modified_chamfer(self, s1, s2):
        dist2 = self._dist2(s1, s2)
        dist = self.sqrt(dist2)
        return self._unidirectional_chamfer(dist, reverse=False) + \
            self._unidirectional_chamfer(dist, reverse=True)

    def bidirectional_modified_chamfer(self, s1, s2):
        self._size_check(s1, s2)
        return self._bidirectional_modified_chamfer(s1, s2)

    def modified_chamfer(self, s1, s2):
        return self.bidirectional_modified_chamfer(s1, s2)
