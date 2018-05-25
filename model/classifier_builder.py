from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from shapenet.core import cat_desc_to_id, cat_id_to_desc
from .builder import ModelBuilder
from .template_ffd_builder import get_mobilenet_features
from .template_ffd_builder import batch_norm_then

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


def get_tf_dataset(
        render_config, view_index, cat_ids, example_ids, num_parallel_calls=8,
        shuffle=False, repeat=False, batch_size=None):
    from .data import get_image_dataset
    dids_ds = get_image_dataset(
        cat_ids, example_ids, view_index, render_config)
    dids_ds.open()

    cat_indices, example_ids, view_indices = zip(*dids_ds.keys())

    n_examples = len(view_indices)
    cat_indices = tf.convert_to_tensor(cat_indices, tf.int32)
    example_ids = tf.convert_to_tensor(example_ids, tf.string)
    view_indices = tf.convert_to_tensor(view_indices, tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (cat_indices, example_ids, view_indices))

    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(n_examples)

    def map_fn_np(cat_index, example_id, view_index):
        return dids_ds[cat_index, example_id, view_index]

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
        # mode = tf.estimator.ModeKeys.TRAIN
        mobilenet_features = get_mobilenet_features(
            image, mode, load_weights, alpha)
        final_filters = self.params.get('final_conv_filters')
        if final_filters is None:
            final_features = tf.reduce_mean(mobilenet_features, axis=(1, 2))
        else:
            activation = batch_norm_then(
                tf.nn.relu6, training=mode == tf.estimator.ModeKeys.TRAIN)
            final_features = tf.layers.conv2d(
                mobilenet_features, final_filters, 1, 1, activation=activation)
            final_features = tf.layers.flatten(final_features)

        logits = tf.layers.dense(final_features, self.n_classes)
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
        probs = prediction_data['probs']
        pred = prediction_data['predictions']
        plt.imshow(image)
        cat_descs = self.cat_descs
        for cat_desc, prob in zip(cat_descs, probs):
            print('%.3f: %s' % (prob, cat_desc))
        plt.title('%s, inferred %s'
                  % (cat_descs[label_data], cat_descs[pred]))
        plt.show()

    def get_predictions(self, inferences):
        preds = inferences.copy()
        logits = inferences['logits']
        preds['predictions'] = tf.argmax(logits, axis=-1)
        preds['probs'] = tf.nn.softmax(logits)
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
