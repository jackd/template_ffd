import os
import util3d.voxel.dataset as bvd
from shapenet.core.voxels.config import VoxelConfig

_voxels_dir = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), '_filled_voxels')


def fill_voxels(voxels):
    import nuympy as np
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


def _get_filled_gt_voxel_dataset(cat_id, mode):
    folder = os.path.join(_voxels_dir, cat_id)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return bvd.BinvoxDataset(folder, mode=mode)


def create_filled_gt_data(cat_id, example_ids=None, overwrite=False):
    from ids import get_example_ids
    unfilled = get_unfilled_gt_voxel_dataset(cat_id)
    if example_ids is None:
        example_ids = get_example_ids(cat_id, 'eval')
    unfilled = unfilled.subset(example_ids)
    dst = _get_filled_gt_voxel_dataset(cat_id, 'a')
    with unfilled:
        with dst:
            create_filled_data(
                unfilled, dst,
                overwrite=overwrite,
                message='Filling ground truth voxels...')


def get_filled_gt_voxel_dataset(cat_id, auto_save=True, example_ids=None):
    if auto_save:
        create_filled_gt_data(cat_id, example_ids)
    return _get_filled_gt_voxel_dataset(cat_id, 'r')


def get_unfilled_gt_voxel_dataset(cat_id):
    config = VoxelConfig()
    return config.get_dataset(cat_id)


def get_gt_voxel_dataset(
        cat_id, filled=False, auto_save=True, example_ids=None):
    kwargs = dict(auto_save=auto_save, example_ids=example_ids)
    if filled:
        return get_filled_gt_voxel_dataset(cat_id, **kwargs)
    else:
        return get_unfilled_gt_voxel_dataset(cat_id, **kwargs)
