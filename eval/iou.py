from __future__ import division
import numpy as np
from dids import Dataset
from dids.file_io.json_dataset import JsonAutoSavingManager
from shapenet.core.voxels.config import VoxelConfig
from template_ffd.inference.voxels import get_voxel_dataset
from template_ffd.data.ids import get_example_ids
from template_ffd.model import load_params
from template_ffd.data.voxels import get_gt_voxel_dataset
from shapenet.core import cat_desc_to_id
from path import get_eval_path
from template_ffd.model import get_builder


def intersection_over_union(v0, v1):
    intersection = np.sum(np.logical_and(v0, v1))
    union = np.sum(np.logical_or(v0, v1))
    return intersection / union


class IouTemplateSavingManager(JsonAutoSavingManager):
    def __init__(self, model_id, filled=True, voxel_config=None):
        self._model_id = model_id
        self._filled = filled
        self._voxel_config = VoxelConfig() if voxel_config is None else \
            voxel_config

    @property
    def saving_message(self):
        return ('Creating selected template IoU data\nmodel_id: %s\nfilled: %s'
                % (self._model_id, self._filled))

    @property
    def path(self):
        fs = 'filled' if self._filled else 'unfilled'
        return get_eval_path(
            'iou', 'template',
            self._voxel_config.voxel_id, fs, '%s.json' % self._model_id)

    def get_lazy_dataset(self):
        from template_ffd.inference.predictions import \
            get_selected_template_idx_dataset
        builder = get_builder(self._model_id)
        template_ids = builder.template_ids

        gt_ds = get_gt_voxel_dataset(
            builder.cat_id, filled=self._filled, auto_save=True,
            example_ids=template_ids)
        gt_ds = gt_ds.map(lambda v: v.data)
        with gt_ds:
            template_voxels = tuple(gt_ds[tid] for tid in template_ids)

        selected_ds = get_selected_template_idx_dataset(self._model_id)
        selected_ds = selected_ds.map(lambda i: template_voxels[i])

        return Dataset.zip(selected_ds, gt_ds).map(
            lambda v: intersection_over_union(*v))


class IouAutoSavingManager(JsonAutoSavingManager):
    def __init__(
            self, model_id, edge_length_threshold=0.1, filled=False,
            voxel_config=None):
        self._model_id = model_id
        self._edge_length_threshold = edge_length_threshold
        self._filled = filled
        self._voxel_config = VoxelConfig() if voxel_config is None else \
            voxel_config

    @property
    def saving_message(self):
        return ('Creating IoU data\n'
                'model_id: %s\n'
                'edge_length_threshold: %.3f\n'
                'filled: %s\n'
                'voxel_config: %s' % (
                    self._model_id, self._edge_length_threshold, self._filled,
                    self._voxel_config.voxel_id))

    @property
    def path(self):
        fs = 'filled' if self._filled else 'unfilled'
        return get_eval_path(
            'iou', str(self._edge_length_threshold),
            self._voxel_config.voxel_id, fs, '%s.json' % self._model_id)

    def get_lazy_dataset(self):
        cat_id = cat_desc_to_id(load_params(self._model_id)['cat_desc'])
        example_ids = get_example_ids(cat_id, 'eval')
        inferred_dataset = get_voxel_dataset(
            self._model_id, self._edge_length_threshold, self._voxel_config,
            filled=self._filled, example_ids=example_ids)

        gt_dataset = get_gt_voxel_dataset(cat_id, filled=self._filled)

        voxel_datasets = Dataset.zip(inferred_dataset, gt_dataset)
        voxel_datasets = voxel_datasets.subset(example_ids)

        def map_fn(v):
            return intersection_over_union(v[0].data, v[1].data)

        iou_dataset = voxel_datasets.map(map_fn)
        return iou_dataset


def get_iou_dataset(model_id, edge_length_threshold=0.1, filled=False):
    from shapenet.core import cat_desc_to_id
    from template_ffd.data.ids import get_example_ids
    from template_ffd.model import load_params
    cat_id = cat_desc_to_id(load_params(model_id)['cat_desc'])
    n_eval = len(get_example_ids(cat_id, 'eval'))

    manager = IouAutoSavingManager(
        model_id=model_id,
        edge_length_threshold=edge_length_threshold,
        filled=filled
    )
    with manager.get_saving_dataset() as ds:
        needs_calc = len(ds) < n_eval
    if needs_calc:
        manager.save_all()
    return manager.get_saving_dataset()


def report_iou_average(model_id, edge_length_threshold=0.1, filled=False):
    with get_iou_dataset(model_id, edge_length_threshold=edge_length_threshold,
                         filled=filled) as ds:
        values = list(ds.values())
    print(np.mean(values))
