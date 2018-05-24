from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from shapenet.core import cat_desc_to_id, cat_id_to_desc
from .builder import ModelBuilder
from .template_ffd_builder import get_mobilenet_features

_cat_descs5 = (
    'plane',
    'bench',
    'car',
    'chair',
    'sofa'
)

_cat_descs8 = (
    'cabinet',
    'monitor',
    'lamp',
    'speaker',
    'pistol',
    'table',
    'cellphone',
    'watercraft',
)

_cat_descs13 = _cat_descs5 + _cat_descs8
_cat_ids13 = tuple(cat_desc_to_id(c) for c in _cat_descs13)


def get_dids_dataset(render_config, view_index, cat_ids, example_ids):
    from dids.core import BiKeyDataset
    if not isinstance(view_index, (list, tuple)):
        raise NotImplementedError()
    datasets = {
        i: render_config.get_multi_view_dataset(c, eids)
        for i, (c, eids) in enumerate(zip(cat_ids, example_ids))}
    dataset = BiKeyDataset(datasets)
    return dataset


def get_tf_dataset(
        render_config, view_index, cat_ids, example_ids, num_parallel_calls=8,
        shuffle=False, repeat=False, batch_size=None):
    from shapenet.image import with_background
    dids_ds = get_dids_dataset(render_config, view_index, cat_ids, example_ids)
    dids_ds.open()

    cat_indices = []
    example_ids = []
    view_indices = []
    for cat_index, (example_id, view_index) in dids_ds.keys():
        cat_indices.append(cat_index)
        example_ids.append(example_id)
        view_indices.append(view_index)

    n_examples = len(view_indices)
    cat_indices = tf.convert_to_tensor(cat_indices, tf.int32)
    example_ids = tf.convert_to_tensor(example_ids, tf.string)
    view_indices = tf.convert_to_tensor(view_indices, tf.int32)

    # cat_indices, example_ids = zip(*dids_ds.keys())
    #
    # n_views = len(view_index)
    # n_ids = len(cat_indices)
    # n_examples = n_views * n_ids
    # cat_indices = tf.convert_to_tensor(cat_indices, dtype=tf.int32)
    # example_ids = tf.convert_to_tensor(example_ids, dtype=tf.string)
    #
    # cat_indices = tf.tile(cat_indices, (n_views,))
    # example_ids = tf.tile(example_ids, (n_views,))
    # view_indices = tf.range(n_ids, dtype=tf.int32)
    # view_indices = tf.tile(
    #     tf.expand_dims(view_indices, axis=-1), (1, n_ids))
    # view_indices = tf.reshape(view_indices, (-1,))

    dataset = tf.data.Dataset.from_tensor_slices(
        (cat_indices, example_ids, view_indices))

    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(n_examples)

    def map_fn_np(cat_index, example_id, view_index):
        return with_background(
            np.array(dids_ds[cat_index, (example_id, view_index)]), 255)

    def map_fn_tf(cat_index, example_id, view_index):
        image = tf.py_func(
            map_fn_np, (cat_index, example_id, view_index), tf.uint8)
        image.set_shape(render_config.shape + (3,))
        labels = cat_index
        image = tf.image.per_image_standardization(image)
        features = dict(
            image=image,
            example_id=example_id,
            view_index=view_index,
            cat_index=cat_index
        )
        return features, labels

    dataset = dataset.map(map_fn_tf, num_parallel_calls=num_parallel_calls)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(2)
    return dataset


class ClassifierBuilder(ModelBuilder):
    @property
    def n_classes(self):
        return len(self.cat_ids)

    def get_inference(self, features, mode):
        alpha = self.params.get('alpha', 0.25)
        load_weights = self._initializer_run
        image = features['image']
        mobilenet_features = get_mobilenet_features(
            image, mode, load_weights, alpha)
        pooled_features = tf.reduce_mean(mobilenet_features, axis=(1, 2))
        logits = tf.layers.dense(pooled_features, self.n_classes)
        return dict(
            logits=logits,
            cat_index=features['cat_index'],
            exmaple_id=features['example_id'],
            view_index=features['view_index'])

    def get_inference_loss(self, inference, labels):
        """Get the loss assocaited with inferences."""
        logits = inference['logits']
        return tf.losses.sparse_softmax_cross_entropy(labels, logits)

    def get_train_op(self, loss, step):
        optimizer = tf.train.AdamOptimizer(
            self.params.get('learning_rate', 1e-3))
        return optimizer.minimize(loss, global_step=step)

    @property
    def cat_descs(self):
        if not hasattr(self, '_cat_descs'):
            self._cat_descs = [cat_id_to_desc(c) for c in self.cat_ids]
        return self._cat_descs

    @property
    def cat_ids(self):
        return self.params.get('cat_ids', _cat_ids13)

    def vis_example_data(self, feature_data, label_data):
        import matplotlib.pyplot as plt
        image = feature_data['image']
        image -= np.min(image)
        image /= np.max(image)
        plt.imshow(image)
        plt.title(self.cat_descs[label_data])
        plt.show()

    def vis_prediction_data(self, prediction_data, feature_data, label_data):
        import matplotlib.pyplot as plt
        image = feature_data['image']
        image -= np.min(image)
        image /= np.max(image)
        plt.show(image)
        cat_descs = self.cat_descs
        plt.title('%s, inferred %s'
                  % (cat_descs[prediction_data['predictions']],
                     cat_descs[label_data]))
        plt.show()

    def get_predictions(self, inferences):
        preds = inferences.copy()
        preds['predictions'] = tf.argmax(inferences['logits'], axis=-1)
        return preds

    def get_eval_metric_ops(self, predictions, labels):
        accuracy = tf.metrics.accuracy(
            predictions=predictions['predictions'], labels=labels)
        return dict(accuracy=accuracy)

    @property
    def batch_size(self):
        return 64

    def get_inputs(self, mode):
        from shapenet.core.blender_renderings.config import RenderConfig
        from ..data.ids import get_example_ids
        render_config = RenderConfig()
        view_index = self.params.get(
            'view_index', range(render_config.n_images))
        cat_ids = self.cat_ids
        example_ids = tuple(
            get_example_ids(cat_id, mode) for cat_id in cat_ids)
        dataset = get_tf_dataset(
            render_config, view_index, cat_ids, example_ids,
            batch_size=self.batch_size, shuffle=True,
            repeat=mode == tf.estimator.ModeKeys.TRAIN)
        return dataset.make_one_shot_iterator().get_next()
