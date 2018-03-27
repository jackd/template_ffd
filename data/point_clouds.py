import numpy as np
import tensorflow as tf
from manager import MapManager, base_dataset
from util3d.point_cloud import sample_points
from shapenet.core.point_clouds import get_point_cloud_dataset


class SampledPointCloudManager(MapManager):
    def __init__(self, cat_id, n_samples, n_resamples):
        self._cat_id = cat_id
        self._n_samples = n_samples
        self._n_resamples = n_resamples
        self._dataset = get_point_cloud_dataset(cat_id, n_samples)
        self._dataset.open()

    @property
    def output_shape(self):
        return (self._n_resamples, 3)

    @property
    def output_type(self):
        return tf.float32

    def map_np(self, example_id):
        points = np.array(self._dataset[example_id], dtype=np.float32)
        return sample_points(points, self._n_resamples, axis=0)


def get_sampled_point_cloud_dataset(
        cat_id, example_ids, n_samples, n_resamples):
    manager = SampledPointCloudManager(cat_id, n_samples, n_resamples)
    base = base_dataset(example_ids)
    return base.map(manager.map_tf)


if __name__ == '__main__':
    from mayavi import mlab
    from util3d.mayavi_vis import vis_point_cloud
    from shapenet.core import cat_desc_to_id, get_example_ids
    cat_desc = 'plane'
    n_samples = 16384
    # n_resamples = None
    n_resamples = 1024
    cat_id = cat_desc_to_id(cat_desc)
    example_ids = get_example_ids(cat_id)
    dataset = get_sampled_point_cloud_dataset(
        cat_id, example_ids, n_samples, n_resamples)
    pc = dataset.make_one_shot_iterator().get_next()
    with tf.train.MonitoredSession() as sess:
        while not sess.should_stop():
            cloud = sess.run(pc)
            vis_point_cloud(cloud, color=(0, 0, 1), scale_factor=0.01)
            mlab.show()
