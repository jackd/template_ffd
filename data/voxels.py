import os
import util3d.voxel.dataset as bvd
from shapenet.core.voxels.config import VoxelConfig
from dids.core import BiKeyDataset

_voxels_dir = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), '_filled_voxels')


def fill_voxels(voxels):
    import numpy as np
    from util3d.voxel.manip import filled_voxels
    from util3d.voxel.binvox import DenseVoxels
    if isinstance(voxels, np.ndarray):
        return filled_voxels(voxels)
    else:
        return DenseVoxels(
            filled_voxels(voxels.dense_data()), voxels.translate, voxels.scale)


def create_filled_data(unfilled_dataset, dst, overwrite=False, message=None):
    src = unfilled_dataset.map(fill_voxels)
    with src:
        dst.save_dataset(src, overwrite=overwrite, message=message)


def _get_filled_gt_voxel_dataset_single(cat_id, mode):
    folder = os.path.join(_voxels_dir, cat_id)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return bvd.BinvoxDataset(folder, mode=mode)


def _get_filled_gt_voxel_dataset(cat_id, mode):
    if isinstance(cat_id, (list, tuple)):
        datasets = {k: _get_filled_gt_voxel_dataset_single(k, mode)
                    for k in cat_id}
        return BiKeyDataset(datasets)
    else:
        return _get_filled_gt_voxel_dataset_single(cat_id, mode)


def create_filled_gt_data(cat_id, overwrite=False):
    src = get_unfilled_gt_voxel_dataset(cat_id)
    dst = _get_filled_gt_voxel_dataset(cat_id, 'a')
    with src:
        with dst:
            create_filled_data(
                src, dst, overwrite=overwrite,
                message='Filling ground truth voxels...')


def get_filled_gt_voxel_dataset(cat_id, auto_save=True, example_ids=None):
    if auto_save:
        create_filled_gt_data(cat_id, example_ids)
    return _get_filled_gt_voxel_dataset(cat_id, 'r')


def _get_unfilled_gt_voxel_dataset_single(cat_id):
    return VoxelConfig().get_dataset(cat_id)


def get_unfilled_gt_voxel_dataset(cat_id):
    if isinstance(cat_id, (list, tuple)):
        datasets = {
            k: _get_unfilled_gt_voxel_dataset_single(k) for k in cat_id}
        return BiKeyDataset(datasets)
    else:
        return _get_unfilled_gt_voxel_dataset_single(cat_id)


def get_gt_voxel_dataset(
        cat_id, filled=False, auto_save=True, example_ids=None):
    kwargs = dict(auto_save=auto_save, example_ids=example_ids)
    if filled:
        return get_filled_gt_voxel_dataset(cat_id, **kwargs)
    else:
        return get_unfilled_gt_voxel_dataset(cat_id, **kwargs)
