import os
import numpy as np
import tensorflow as tf
import builder
from shapenet.core import cat_desc_to_id
from template_ffd.metrics.tf_impl import tf_metrics
from template_ffd.templates.ids import get_template_ids
from template_ffd.templates.ffd import get_ffd_dataset
from template_ffd.data.ids import get_example_ids
from template_ffd.templates.mesh import get_template_mesh_dataset


def get_nn(data, query_points):
    from scipy.spatial import cKDTree
    tree = cKDTree(data)
    return tree.query(query_points)[1]


def get_centroids(vertices, faces):
    return np.mean(vertices[faces], axis=-2)


def segment_faces(vertices, faces, points, labels):
    from shapenet.core.annotations import segment
    centroids = get_centroids(vertices, faces)
    i0 = get_nn(points, centroids)
    face_labels = labels[i0]
    assert(len(face_labels) == len(faces))
    segmented_faces = segment(faces, face_labels)
    return segmented_faces


def sample_tf(x, n_resamples, axis=0, name=None):
    n_original = x.shape[axis]
    indices = tf.random_uniform(
        shape=(n_resamples,), minval=0, maxval=n_original, dtype=np.int32)
    return tf.gather(x, indices, axis=axis, name=name)


def batch_norm_then(activation, **bn_kwargs):
    def f(x):
        return activation(tf.layers.batch_normalization(x, **bn_kwargs))
    return f


def get_mobilenet_features(image, mode, load_weights=False, alpha=1):
    from mobilenet import MobileNet
    training = mode == tf.estimator.ModeKeys.TRAIN
    tf.keras.backend.set_learning_phase(training)
    weights = 'imagenet' if load_weights else None

    model = MobileNet(
        input_shape=image.shape.as_list()[1:],
        input_tensor=image,
        include_top=False,
        weights=weights,
        alpha=alpha)
    return model.output


def linear_annealing_factor(cutoff):
    step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
    return tf.maximum(1 - step / cutoff, 0)


def exp_annealing_factor(rate):
    step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
    return tf.exp(-step*rate)


def annealed_weight(
        weight, linear_annealing_cutoff=None, exp_annealing_rate=None):
    if linear_annealing_cutoff is None:
        if exp_annealing_rate is None:
            return weight
        return weight*exp_annealing_factor(exp_annealing_rate)
    else:
        if exp_annealing_rate is None:
            return weight*linear_annealing_factor(linear_annealing_cutoff)
        else:
            raise ValueError(
                'At least one of `linear_annealing_cutoff` or '
                '`exp_annealing_rate` must be `None`')


def get_image_dataset(render_config, cat_id, view_index):
    from shapenet.image import with_background
    import random
    if isinstance(view_index, int):
        dataset = render_config.get_dataset(cat_id, view_index)
    elif isinstance(view_index, (list, tuple)):
        dataset = render_config.get_multi_view_dataset(cat_id)

        def key_fn(example_id):
            return example_id, random.sample(view_index, 1)[0]

        dataset = dataset.map_keys(key_fn)
    else:
        raise TypeError('view_index must be an int or list/tuple of ints')
    return dataset.map(lambda x: with_background(x, 255))


def get_cloud_dataset(cat_id, n_samples, n_resamples):
    from shapenet.core.point_clouds import PointCloudAutoSavingManager
    from util3d.point_cloud import sample_points
    manager = PointCloudAutoSavingManager(cat_id, n_samples)
    if not os.path.isfile(manager.path):
        manager.save_all()
    dataset = manager.get_saving_dataset(mode='r')
    # get_saved_dataset seems to give issues because we don't close properly
    # dataset = manager.get_saved_dataset()
    return dataset.map(
        lambda x: sample_points(np.array(x, dtype=np.float32), n_resamples))


def get_dataset(
        render_config, view_index, n_samples, n_resamples, cat_id,
        example_ids, num_parallel_calls=8, shuffle=False, repeat=False,
        batch_size=None):

    image_ds = get_image_dataset(render_config, cat_id, view_index)
    image_ds.open()
    if not all(k in image_ds for k in example_ids):
        raise KeyError('Not all images present')
    cloud_ds = get_cloud_dataset(cat_id, n_samples, n_resamples)
    cloud_ds.open()
    if not all(k in cloud_ds for k in example_ids):
        raise KeyError('Not all cloud data present')

    def map_np(example_id):
        return image_ds[example_id], cloud_ds[example_id]

    def map_tf(example_id):
        image, cloud = tf.py_func(
            map_np, [example_id], (tf.uint8, tf.float32), stateful=False)
        image.set_shape(tuple(render_config.shape) + (3,))
        cloud.set_shape((n_resamples, 3))
        image = tf.image.per_image_standardization(image)
        return example_id, image, cloud

    dataset = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(example_ids, tf.string))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(example_ids))
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.map(
        map_tf, num_parallel_calls=num_parallel_calls)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(2)
    # dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))

    return dataset


class TemplateFfdBuilder(builder.ModelBuilder):
    def __init__(self, *args, **kwargs):
        super(TemplateFfdBuilder, self).__init__(*args, **kwargs)
        self._initializer_run = False

    @property
    def n_ffd_samples(self):
        return self.params.get('n_ffd_samples', 16384)

    @property
    def view_index(self):
        return self.params.get('view_index', 5)

    def _get_ffd_data(self, ffd_dataset):
        for example_id in self.template_ids:
            ffd_data = ffd_dataset[example_id]
            b, p = (np.array(ffd_data[k]) for k in ('b', 'p'))
            yield example_id, b, p

    def get_ffd_data(self, ffd_dataset=None):
        if ffd_dataset is None:
            n_ffd_points = self.n_ffd_samples
            ffd_dataset = get_ffd_dataset(
                self.cat_id, self.n, n_samples=n_ffd_points)
            with ffd_dataset:
                return tuple(self._get_ffd_data(ffd_dataset))
        else:
            return self._get_ffd_data(ffd_dataset)

    def get_ffd_tensors(self, ffd_dataset=None):
        n_ffd_resamples = self.params.get('n_ffd_resamples', 1024)
        bs = []
        ps = []
        for example_id, b, p in self.get_ffd_data(ffd_dataset):
            b = tf.constant(b, dtype=tf.float32)
            b = sample_tf(
                b, n_ffd_resamples, axis=0, name='b_resampled_%s' % example_id)
            bs.append(b)
            ps.append(p)
        b = tf.stack(bs)
        p = tf.constant(np.array(ps), dtype=tf.float32)

        return b, p

    def get_image_features(self, image, mode, **inference_params):
        alpha = inference_params.get('alpha', 1)
        load_weights = self._initializer_run
        features = get_mobilenet_features(image, mode, load_weights, alpha)
        conv_filters = inference_params.get('final_conv_filters', [64])
        activation = batch_norm_then(
            tf.nn.relu6, training=mode == tf.estimator.ModeKeys.TRAIN)
        for n in conv_filters:
            features = tf.layers.conv2d(
                features, n, 1, activation=activation)
            features = tf.layers.batch_normalization(features)
        return features

    def get_inference(self, features, mode):
        """Get inferred value of the model."""
        inference_params = self.params.get('inference_params', {})
        training = mode == tf.estimator.ModeKeys.TRAIN
        image = features['image']
        example_id = features['example_id']
        features = self.get_image_features(image, mode, **inference_params)
        features = tf.layers.flatten(features)

        for n_dense in inference_params.get('final_dense_nodes', [512]):
            features = tf.layers.dense(
                features, n_dense, activation=batch_norm_then(
                    tf.nn.relu6, training=training))

        n_control_points = self.n_control_points
        n_templates = self.n_templates

        dp = tf.layers.dense(
            features, n_templates * n_control_points * 3,
            kernel_initializer=tf.random_normal_initializer(stddev=1e-4))
        dp = tf.reshape(dp, (-1, n_templates, n_control_points, 3))
        probs = tf.layers.dense(
            features, n_templates, activation=tf.nn.softmax)
        eps = self.params.get('prob_eps', 0.1)
        if eps > 0:
            probs = (1 - eps)*probs + eps / n_templates
        return dict(example_id=example_id, probs=probs, dp=dp)

    @property
    def cat_id(self):
        return cat_desc_to_id(self.params['cat_desc'])

    @property
    def n_templates(self):
        return len(self.template_ids)

    @property
    def n_control_points(self):
        return (self.n + 1)**3

    @property
    def template_ids(self):
        template_ids = get_template_ids(self.cat_id)
        idxs = self.params.get('template_idxs')
        if idxs:
            template_ids = tuple(template_ids[i] for i in idxs)
        return template_ids

    def get_inferred_point_clouds(self, dp):
        b, p = self.get_ffd_tensors()
        inferred_point_clouds = tf.einsum('ijk,likm->lijm', b, p + dp)
        return inferred_point_clouds

    def get_chamfer_loss(self, gamma, dp, ground_truth_cloud):
        inferred_point_clouds = self.get_inferred_point_clouds(dp)
        inferred_point_clouds = tf.unstack(inferred_point_clouds, axis=1)
        losses = [tf_metrics.chamfer(inferred, ground_truth_cloud)
                  for inferred in inferred_point_clouds]
        losses = tf.stack(losses, axis=1)
        losses = gamma * losses
        loss = tf.reduce_sum(losses)
        return loss

    def get_entropy_loss(self, probs, **weight_kwargs):
        mean_probs = tf.reduce_mean(probs, axis=0)  # average across batch
        entropy_loss = tf.reduce_sum(mean_probs * tf.log(mean_probs))
        weight = annealed_weight(**weight_kwargs)
        return entropy_loss * weight

    def get_dp_reg_loss(self, probs, dp, **weight_kwargs):
        if weight_kwargs.pop('uniform', False):
            reg_loss = tf.reduce_sum(dp**2)
        else:
            reg_loss = tf.reduce_sum(dp**2, axis=(2, 3))
            reg_loss *= probs
            reg_loss = tf.reduce_sum(reg_loss)
        weight = annealed_weight(**weight_kwargs)
        return reg_loss*weight

    def get_inference_loss(self, inference, labels):
        """Get the loss assocaited with inferences."""
        probs, dp = (inference[k] for k in ('probs', 'dp'))
        ground_truth_cloud = labels
        losses = []

        gamma_code = self.params.get('gamma', 'linear')
        if gamma_code == 'linear':
            gamma = probs
        elif gamma_code == 'square':
            gamma = probs ** 2
        elif gamma_code == 'log':
            gamma = -tf.log(1 - probs)
        else:
            raise ValueError(
                'Unrecognized gamma value in params: %s' % gamma_code)
        chamfer_loss = self.get_chamfer_loss(gamma, dp, ground_truth_cloud)
        tf.summary.scalar('chamfer', chamfer_loss, family='sublosses')
        losses.append(chamfer_loss)

        entropy_params = self.params.get('entropy_loss')
        if entropy_params is not None:
            entropy_loss = self.get_entropy_loss(probs, **entropy_params)
            tf.summary.scalar('entropy', entropy_loss, family='sublosses')
            losses.append(entropy_loss)

        dp_reg_params = self.params.get('dp_regularization')
        if dp_reg_params is not None:
            dp_reg_loss = self.get_dp_reg_loss(probs, dp, **dp_reg_params)
            tf.summary.scalar('dp_reg_loss', dp_reg_loss, family='sublosses')
            losses.append(dp_reg_loss)

        loss = losses[0] if len(losses) == 1 else tf.add_n(losses)

        return loss

    def get_train_op(self, loss, step):
        """Get the train operation."""
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.params.get('learning_rate', 1e-3))
        return optimizer.minimize(loss, step)

    @property
    def batch_size(self):
        return self.params.get('batch_size', 32)

    @property
    def n(self):
        return self.params.get('n', 3)

    @property
    def render_config(self):
        from shapenet.core.blender_renderings.config import RenderConfig
        return RenderConfig(**self.params.get('render_params', {}))

    @property
    def n_samples(self):
        return self.params.get('n_samples', 16384)

    def get_dataset(self, mode):
        cat_id = self.cat_id
        example_ids = get_example_ids(cat_id, mode)
        render_config = self.render_config
        view_index = self.view_index
        n_samples = self.n_samples
        n_resamples = self.params.get('n_resamples', 1024)
        repeat = mode == tf.estimator.ModeKeys.TRAIN
        shuffle = repeat
        batch_size = self.batch_size

        dataset = get_dataset(
            render_config, view_index, n_samples, n_resamples, cat_id,
            example_ids, shuffle=shuffle, repeat=repeat, batch_size=batch_size)
        return dataset

    def get_inputs(self, mode):
        dataset = self.get_dataset(mode)
        # return dataset.make_one_shot_iterator().get_next()
        example_id, image, cloud = dataset.make_one_shot_iterator().get_next()
        return dict(example_id=example_id, image=image), cloud

    def vis_example_data(self, feature_data, label_data):
        import matplotlib.pyplot as plt
        from util3d.mayavi_vis import vis_point_cloud
        from mayavi import mlab
        image = feature_data['image']
        point_cloud = label_data
        image -= np.min(image)
        image /= np.max(image)
        plt.imshow(image)
        plt.show(block=False)
        vis_point_cloud(point_cloud, color=(0, 0, 1), scale_factor=0.01)
        mlab.show()
        plt.close()

    def get_prediction_to_mesh_fn(self, edge_length_threshold=None):
        cat_id = self.cat_id
        with get_ffd_dataset(cat_id, self.n, edge_length_threshold) \
                as ffd_dataset:
            example_ids, bs, ps = zip(*self.get_ffd_data(ffd_dataset))
        with get_template_mesh_dataset(cat_id, edge_length_threshold) as \
                mesh_dataset:
            all_faces = []
            all_vertices = []
            for k in example_ids:
                sg = mesh_dataset[k]
                all_faces.append(np.array(sg['faces']))
                all_vertices.append(np.array(sg['vertices']))

        def transform_predictions(probs, dp):
            i = np.argmax(probs)
            vertices = np.matmul(bs[i], ps[i] + dp[i])
            faces = all_faces[i]
            original_vertices = all_vertices[i]
            return dict(
                vertices=vertices,
                faces=faces,
                original_vertices=original_vertices,
                attrs=dict(template_id=example_ids[i]))

        return transform_predictions

    def get_prediction_to_top_k_mesh_fn(
            self, edge_length_threshold=None, top_k=2):
        cat_id = self.cat_id
        with get_ffd_dataset(cat_id, self.n, edge_length_threshold) \
                as ffd_dataset:
            example_ids, bs, ps = zip(*self.get_ffd_data(ffd_dataset))
        with get_template_mesh_dataset(cat_id, edge_length_threshold) as \
                mesh_dataset:
            all_faces = []
            all_vertices = []
            for k in example_ids:
                sg = mesh_dataset[k]
                all_faces.append(np.array(sg['faces']))
                all_vertices.append(np.array(sg['vertices']))

        def get_deformed_mesh(i, dp):
            vertices = np.matmul(bs[i], ps[i] + dp[i])
            faces = all_faces[i]
            return dict(
                vertices=vertices, faces=faces,
                original_vertices=all_vertices[i])

        def transform_predictions(probs, dp):
            ks = probs.argsort()[-3:][::-1]
            return [get_deformed_mesh(k, dp) for k in ks]

        return transform_predictions

    def get_prediction_to_cloud_fn(self, n_samples=None):
        from util3d.point_cloud import sample_points
        with get_ffd_dataset(
                self.cat_id, self.n, n_samples=self.n_ffd_samples) \
                as ffd_dataset:
            example_ids, bs, ps = zip(*self.get_ffd_data(ffd_dataset))

        def transform_predictions(probs, dp):
            i = np.argmax(probs)
            b = bs[i]
            if n_samples is not None:
                b = sample_points(b, n_samples)
            points = np.matmul(b, ps[i] + dp[i])
            return dict(cloud=points, attrs=dict(template_id=example_ids[i]))

        return transform_predictions

    def get_segmented_cloud_fn(self):
        from shapenet.core.annotations.datasets import PointCloudDataset, \
            SegmentationDataset
        import template_ffd.templates.annotations_ffd as ann
        cat_id = self.cat_id
        bs = []
        ps = []
        segs = []
        original_points = []
        with ann.get_annotations_ffd_dataset(cat_id, self.n) as ds:
            for k in self.template_ids:
                if k in ds:
                    subgroup = ds[k]
                    b, p = (np.array(subgroup[kk]) for kk in ('b', 'p'))
                else:
                    b = None
                    p = None
                bs.append(b)
                ps.append(p)

        with SegmentationDataset(cat_id) as sd:
            for k in self.template_ids:
                if k in sd:
                    seg = sd[k]
                else:
                    seg = None
                segs.append(seg)

        with PointCloudDataset(cat_id) as ds:
            for k in self.template_ids:
                if k in ds:
                    points = ds[k]
                else:
                    points = None
                original_points.append(points)

        def transform_predictions(probs, dp):
            i = np.argmax(probs)
            b = bs[i]
            if b is None:
                return None
            else:
                points = np.matmul(b, ps[i] + dp[i])
                return dict(
                    points=points,
                    segmentation=segs[i],
                    original_points=original_points[i]
                )

        return transform_predictions

    def get_segmented_mesh_fn(self, edge_length_threshold=None):
        from shapenet.core.annotations.datasets import PointCloudDataset, \
            SegmentationDataset
        from dataset import Dataset

        cat_id = self.cat_id
        bs = []
        ps = []
        segs = []
        faces = []
        original_segs = []
        original_seg_points = []
        ffd_dataset = get_ffd_dataset(
            cat_id, self.n, edge_length_threshold=edge_length_threshold)
        with ffd_dataset:
            example_ids, bs, ps = zip(*self.get_ffd_data(ffd_dataset))

        template_mesh_ds = get_template_mesh_dataset(
                cat_id, edge_length_threshold=edge_length_threshold)
        seg_points_ds = PointCloudDataset(cat_id)
        seg_ds = SegmentationDataset(cat_id)
        ds = Dataset.zip(template_mesh_ds, seg_points_ds, seg_ds)
        with ds:
            for example_id in example_ids:
                if example_id in ds:
                    template_mesh, seg_points, original_seg = ds[example_id]
                    v, f = (np.array(template_mesh[k])
                            for k in ('vertices', 'faces'))
                    centroids = get_centroids(v, f)
                    seg = original_seg[get_nn(seg_points, centroids)]
                else:
                    f = None
                    seg = None
                    seg_points = None
                    original_seg = None
                segs.append(seg)
                original_seg_points.append(seg_points)
                original_segs.append(original_seg)
                faces.append(f)

        def transform_predictions(probs, dp):
            i = np.argmax(probs)
            seg = segs[i]
            if seg is None:
                return None
            else:
                v = np.matmul(bs[i], ps[i] + dp[i])
                return dict(
                    faces=faces[i], vertices=v, segmentation=segs[i],
                    original_points=original_seg_points[i],
                    original_segmentation=original_segs[i])

        return transform_predictions

    def vis_prediction_data(
            self, prediction_data, feature_data, label_data=None):
        import matplotlib.pyplot as plt
        from util3d.mayavi_vis import vis_mesh
        from mayavi import mlab
        image = feature_data['image']
        dp = prediction_data['dp']
        probs = prediction_data['probs']

        if not hasattr(self, '_mesh_fn') or self._mesh_fn is None:
            self._mesh_fn = self.get_prediction_to_mesh_fn()
        image -= np.min(image)
        image /= np.max(image)
        plt.imshow(image)

        mesh = self._mesh_fn(probs, dp)
        vertices, faces, original_vertices = (
            mesh[k] for k in('vertices', 'faces', 'original_vertices'))
        mlab.figure()
        vis_mesh(
            vertices, faces, color=(0, 1, 0), include_wireframe=False,
            axis_order='xzy')
        mlab.figure()
        vis_mesh(
            original_vertices, faces, color=(1, 0, 0), include_wireframe=False,
            axis_order='xzy')

        plt.show(block=False)
        mlab.show()
        plt.close()
