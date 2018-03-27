import tensorflow as tf
from base import Metrics


class _TensorflowMetrics(Metrics):
    def sum(self, array, axis=None):
        return tf.reduce_sum(array, axis=axis)

    def max(self, array, axis=None):
        return tf.reduce_max(array, axis=axis)

    def min(self, array, axis=None):
        return tf.reduce_min(array, axis=axis)

    def expand_dims(self, array, axis):
        return tf.expand_dims(array, axis=axis)

    def sqrt(self, x):
        return tf.sqrt(x)

    def top_k(self, x, k, axis=-1):
        n = len(x.shape)
        if axis not in [-1, n-1]:
            if axis < 0:
                axis += n
            x = tf.transpose(x, range(axis) + range(axis+1, n) + [axis])
        return tf.nn.top_k(x, k)

    # def _size_check(self, s1, s2):
    #     for s1s, s2s in zip(s1.shape[:-2], s2.shape[:-2]):
    #         if s1s != s2s and s1s is not None and s2s is not None:
    #             raise ValueError('s1 and s2 must share same shape[:-2]')
    #     if s1.shape[-1] != s2.shape[-1]:
    #         raise ValueError(
    #             'last dim of s1 and s2 must be same, but got %d, %d'
    #             % (s1.shape[-1], s2.shape[-1]))

    def _unidirectional_chamfer(self, dist2, reverse=False):
        with tf.name_scope('chamfer_unidirecitonal'):
            return super(_TensorflowMetrics, self)._unidirectional_chamfer(
                dist2, reverse=reverse)

    def _bidirectional_chamfer(self, s1, s2):
        from tf_nearest_neighbour import nn_distance
        with tf.name_scope('chamfer'):
            shape1 = s1.shape.as_list()
            shape2 = s2.shape.as_list()

            s1 = tf.reshape(s1, [-1] + shape1[-2:])
            s2 = tf.reshape(s2, [-1] + shape2[-2:])
            dist1, _, dist2, __ = nn_distance(s1, s2)
            loss1 = tf.reduce_sum(dist1, axis=-1)
            loss2 = tf.reduce_sum(dist2, axis=-1)
            if len(shape1) > 3:
                loss1 = tf.reshape(loss1, shape1[:-2])
            if len(shape2) > 3:
                loss2 = tf.reshape(loss2, shape2[:-2])
            return loss1 + loss2

    def _unidirectional_hausdorff(self, dist2, reverse=False):
        with tf.name_scope('chamfer_unidirecitonal'):
            return super(_TensorflowMetrics, self)._unidirectional_hausdorff(
                dist2, reverse=reverse)

    def _bidirectional_hausdorff(self, dist2, reverse=False):
        with tf.name_scope('hausdorff'):
            return super(_TensorflowMetrics, self)._bidirectional_hausdorff(
                dist2, reverse=reverse)

    def unidirectional_modified_chamfer(self, s1, s2, reverse=False):
        with tf.name_scope('modified_chamfer_unidirectional'):
            return super(
                _TensorflowMetrics, self).unidirectional_modified_chamfer(
                    s1, s2, reverse=reverse)

    def _bidirectional_modified_chamfer(self, s1, s2):
        with tf.name_scope('modified_chamfer'):
            return super(
                _TensorflowMetrics, self)._bidirectional_modified_chamfer(
                    s1, s2)


tf_metrics = _TensorflowMetrics()
