import tensorflow as tf
from shapenet.image import with_background
from manager import MapManager, base_dataset


class RenderingsManager(MapManager):
    def __init__(self, render_config, view_index, cat_id):
        self._config = render_config
        self._dataset = render_config.get_dataset(cat_id, view_index)
        self._dataset.open()
        self._cat_id = cat_id

    def map_np(self, example_id):
        return with_background(self._dataset[example_id], 255)

    @property
    def output_shape(self):
        return self._config.shape + (3,)

    @property
    def output_type(self):
        return tf.uint8


def get_renderings_dataset(
        render_config, view_index, cat_id, example_ids):
    manager = RenderingsManager(render_config, view_index, cat_id)
    return base_dataset(example_ids).map(manager.map_tf)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from shapenet.core import cat_desc_to_id, get_example_ids
    from shapenet.core.blender_renderings.config import RenderConfig
    cat_desc = 'plane'
    view_index = 5
    config = RenderConfig()
    cat_id = cat_desc_to_id(cat_desc)
    example_ids = get_example_ids(cat_id)
    dataset = get_renderings_dataset(config, view_index, cat_id, example_ids)
    image_tf = dataset.make_one_shot_iterator().get_next()
    with tf.train.MonitoredSession() as sess:
        while not sess.should_stop():
            image = sess.run(image_tf)
            plt.imshow(image)
            plt.show()
