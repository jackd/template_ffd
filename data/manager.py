import tensorflow as tf


def base_dataset(example_ids):
    # indices = tf.range(len(example_ids), dtype=tf.int32)
    # example_ids = tf.convert_to_tensor(example_ids, tf.string)
    #
    # def map_fn(index):
    #     return tf.gather(example_ids, index)
    #
    # return tf.data.Dataset.from_tensor_slices(indices).map(map_fn)
    example_ids = tf.convert_to_tensor(example_ids, tf.string)
    return tf.data.Dataset.from_tensor_slices(example_ids)


class MapManager(object):
    @property
    def output_shape(self):
        raise NotImplementedError('Abstract method')

    @property
    def output_type(self):
        raise NotImplementedError('Abstract method')

    def map_np(self, example_id):
        raise NotImplementedError('Abstract method')

    def map_tf(self, example_id):
        return tf.py_func(
            self.map_np, [example_id], self.output_type, stateful=False)

    def get_generator_dataset(self, example_ids):
        def generator_fn():
            for example_id in example_ids:
                yield self.map_np(example_id)

        return tf.data.Dataset.from_generator(
            generator_fn, self.output_type, self.output_shape)


class ZippedMapManager(MapManager):
    def __init__(self, managers):
        self._managers = managers

    @property
    def output_shape(self):
        return tuple(m.output_shape for m in self._managers)

    @property
    def output_type(self):
        return tuple(m.output_type for m in self._managers)

    def map_np(self, example_id):
        return tuple(m.map_np(example_id) for m in self._managers)


if __name__ == '__main__':
    from point_clouds import SampledPointCloudManager
    from renderings import RenderingsManager
    from shapenet.core import cat_desc_to_id, get_example_ids
    from shapenet.core.blender_renderings.config import RenderConfig
    cat_desc = 'plane'
    cat_id = cat_desc_to_id(cat_desc)
    example_ids = get_example_ids(cat_id)

    view_index = 5
    render_config = RenderConfig()
    renderings_manager = RenderingsManager(render_config, view_index, cat_id)

    n_samples = 16384
    n_resamples = 1024
    cloud_manager = SampledPointCloudManager(cat_id, n_samples, n_resamples)

    manager = ZippedMapManager((renderings_manager, cloud_manager))

    dataset = base_dataset(example_ids).map(manager.map_tf)
    image, cloud = dataset.make_one_shot_iterator().get_next()

    def vis(image, cloud):
        import matplotlib.pyplot as plt
        from mayavi import mlab
        from util3d.mayavi_vis import vis_point_cloud
        plt.imshow(image)
        vis_point_cloud(cloud, color=(0, 0, 1), scale_factor=0.01)
        plt.show(block=False)
        mlab.show()
        plt.close()

    with tf.train.MonitoredSession() as sess:
        while not sess.should_stop():
            im, cl = sess.run([image, cloud])
            vis(im, cl)
