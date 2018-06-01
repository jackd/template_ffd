import string
import numpy as np
from dids.file_io.json_dataset import JsonAutoSavingManager
from template_ffd.metrics.np_impl import np_metrics

import template_ffd.inference.clouds as clouds
from template_ffd.model import get_builder
from path import get_eval_path
from point_cloud import get_lazy_evaluation_dataset


def _get_lazy_emd_dataset(inf_cloud_dataset, cat_id, n_samples):
    return get_lazy_evaluation_dataset(
        inf_cloud_dataset, cat_id, n_samples,
        lambda c0, c1: np_metrics.emd(c0, c1))


class _TemplateEmdAutoSavingManager(JsonAutoSavingManager):
    def __init__(self, model_id, n_samples=1024):
        self._model_id = model_id
        self._n_samples = n_samples

    @property
    def path(self):
        return get_eval_path(
            'emd', 'template', str(self._n_samples),
            '%s.json' % self._model_id)

    @property
    def saving_message(self):
        return ('Creating chosen template EMD data\n'
                'model_id: %s\nn_samples: %d' %
                (self._model_id, self._n_samples))

    def get_lazy_dataset(self):
        from shapenet.core.point_clouds import get_point_cloud_dataset
        from util3d.point_cloud import sample_points
        from template_ffd.model import get_builder
        from template_ffd.inference.predictions import get_predictions_dataset
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

        predictions = get_predictions_dataset(
            self._model_id)
        inf_cloud_ds = predictions.map(lambda i: clouds[i].copy())
        return _get_lazy_emd_dataset(inf_cloud_ds, cat_id, self._n_samples)


class _EmdAutoSavingManager(JsonAutoSavingManager):
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
        return 'Creating EMD data\n%s' % string.join(
            ('%s: %s' % (k, v) for k, v in items), '\n')

    def get_inferred_cloud_dataset(self):
        raise NotImplementedError('Abstract method')

    def get_lazy_dataset(self):
        inf_cloud_ds = self.get_inferred_cloud_dataset()
        cat_id = get_builder(self._model_id).cat_id
        return _get_lazy_emd_dataset(inf_cloud_ds, cat_id, self._n_samples)


class _PreSampledEmdAutoSavingManager(_EmdAutoSavingManager):
    @property
    def path(self):
        return get_eval_path(
            'emd', 'presampled', str(self._n_samples),
            '%s.json' % self._model_id)

    def get_inferred_cloud_dataset(self):
        return clouds.get_inferred_cloud_dataset(
            pre_sampled=True, model_id=self._model_id,
            n_samples=self._n_samples, **self._kwargs)


class _PostSampledEmdAutoSavingManager(_EmdAutoSavingManager):
    @property
    def path(self):
        return get_eval_path(
            'emd', 'postsampled', str(self._n_samples),
            '%s.json' % self._model_id)

    def get_inferred_cloud_dataset(self):
        return clouds.get_inferred_cloud_dataset(
            pre_sampled=False, model_id=self._model_id,
            n_samples=self._n_samples, **self._kwargs)


def get_emd_manager(model_id, pre_sampled=True, **kwargs):
    if pre_sampled:
        return _PreSampledEmdAutoSavingManager(model_id, **kwargs)
    else:
        return _PostSampledEmdAutoSavingManager(model_id, **kwargs)


def get_emd_average(model_id, pre_sampled=True, **kwargs):
    import os
    manager = get_emd_manager(model_id, pre_sampled, **kwargs)
    values = None
    if os.path.isfile(manager.path):
        with manager.get_saving_dataset('r') as ds:
                values = np.array(tuple(ds.values()))
    if values is None:
        manager.save_all()
        with manager.get_saving_dataset('r') as ds:
            values = np.array(tuple(ds.values()))
    return np.mean(values)


def get_template_emd_manager(model_id, n_samples=1024):
    return _TemplateEmdAutoSavingManager(model_id, n_samples)
