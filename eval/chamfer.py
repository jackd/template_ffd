import string
import numpy as np
from dids.file_io.json_dataset import JsonAutoSavingManager
from template_ffd.metrics.np_impl import np_metrics

import template_ffd.inference.clouds as clouds
from template_ffd.model import get_builder
from path import get_eval_path
from point_cloud import get_lazy_evaluation_dataset


def _get_lazy_chamfer_dataset(inf_cloud_dataset, cat_id, n_samples):
    return get_lazy_evaluation_dataset(
        inf_cloud_dataset, cat_id, n_samples,
        lambda c0, c1: np_metrics.chamfer(c0, c1) / n_samples)


class _TemplateChamferAutoSavingManager(JsonAutoSavingManager):
    def __init__(self, model_id, n_samples=1024):
        self._model_id = model_id
        self._n_samples = n_samples

    @property
    def path(self):
        return get_eval_path(
            'chamfer', 'template',
            str(self._n_samples),
            '%s.json' % self._model_id)

    @property
    def saving_message(self):
        return ('Creating chosen template Chamfer data\n'
                'model_id: %s\nn_samples: %d' %
                (self._model_id, self._n_samples))

    def get_lazy_dataset(self):
        from shapenet.core.point_clouds import get_point_cloud_dataset
        from util3d.point_cloud import sample_points
        from template_ffd.model import get_builder
        from template_ffd.inference.predictions import \
            get_selected_template_idx_dataset
        builder = get_builder(self._model_id)
        cat_id = builder.cat_id
        template_ids = builder.template_ids
        clouds = []

        def sample_fn(cloud):
            return sample_points(np.array(cloud), self._n_samples)

        gt_clouds = get_point_cloud_dataset(
            cat_id, builder.n_samples).map(sample_fn)
        with gt_clouds:
            for example_id in template_ids:
                clouds.append(np.array(gt_clouds[example_id]))

        idx_dataset = get_selected_template_idx_dataset(self._model_id)
        inf_cloud_ds = idx_dataset.map(lambda i: np.array(clouds[i]))
        return _get_lazy_chamfer_dataset(inf_cloud_ds, cat_id, self._n_samples)


class _ChamferAutoSavingManager(JsonAutoSavingManager):
    def __init__(self, model_id, n_samples=1024, **kwargs):
        self._model_id = model_id
        self._n_samples = n_samples
        self._kwargs = kwargs

    @property
    def saving_message(self):
        items = (
                    ('model_id', self._model_id),
                    ('n_samples', self._n_samples)
                ) + tuple(self._kwargs.items())
        return 'Creating Chamfer data\n%s' % string.join(
            ('%s: %s' % (k, v) for k, v in items), '\n')

    def get_inferred_cloud_dataset(self):
        raise NotImplementedError('Abstract method')

    def get_lazy_dataset(self):
        inf_cloud_ds = self.get_inferred_cloud_dataset()
        cat_id = get_builder(self._model_id).cat_id
        return _get_lazy_chamfer_dataset(inf_cloud_ds, cat_id, self._n_samples)


class _PreSampledChamferAutoSavingManager(_ChamferAutoSavingManager):
    @property
    def path(self):
        return get_eval_path(
            'chamfer', 'presampled',
            str(self._n_samples),
            '%s.json' % self._model_id)

    def get_inferred_cloud_dataset(self):
        return clouds.get_inferred_cloud_dataset(
            pre_sampled=True, model_id=self._model_id,
            n_samples=self._n_samples, **self._kwargs)


class _PostSampledChamferAutoSavingManager(_ChamferAutoSavingManager):
    @property
    def path(self):
        return get_eval_path(
            'chamfer', 'postsampled', str(self._n_samples),
            str(self._kwargs['edge_length_threshold']),
            '%s.json' % self._model_id)

    def get_inferred_cloud_dataset(self):
        return clouds.get_inferred_cloud_dataset(
            pre_sampled=False, model_id=self._model_id,
            n_samples=self._n_samples, **self._kwargs)


def get_chamfer_manager(model_id, pre_sampled=True, **kwargs):
    if pre_sampled:
        return _PreSampledChamferAutoSavingManager(model_id, **kwargs)
    else:
        return _PostSampledChamferAutoSavingManager(model_id, **kwargs)


def report_chamfer_average(model_id, pre_sampled=True, **kwargs):
    import os
    from shapenet.core import cat_desc_to_id
    from template_ffd.data.ids import get_example_ids
    from template_ffd.model import load_params
    manager = get_chamfer_manager(model_id, pre_sampled, **kwargs)
    cat_id = cat_desc_to_id(load_params(model_id)['cat_desc'])
    n_eval = len(get_example_ids(cat_id, 'eval'))
    values = None
    if os.path.isfile(manager.path):
        with manager.get_saving_dataset('r') as ds:
            if len(ds) == n_eval:
                values = np.array(tuple(ds.values()))
    if values is None:
        manager.save_all()
        with manager.get_saving_dataset('r') as ds:
            values = np.array(tuple(ds.values()))
    print(np.mean(values))


def get_template_chamfer_manager(model_id, n_samples=1024):
    return _TemplateChamferAutoSavingManager(model_id, n_samples)
