import os
from path import get_inference_subdir
import util3d.voxel.dataset as bvd
import util3d.voxel.convert as bio
from shapenet.core.voxels.config import VoxelConfig
from meshes import get_inferred_mesh_dataset

_default_config = VoxelConfig()


def get_voxel_subdir(model_id, edge_length_threshold=0.1, voxel_config=None,
                     filled=False):
    if voxel_config is None:
        voxel_config = _default_config
    es = 'base' if edge_length_threshold is None else \
        str(edge_length_threshold)
    fs = 'filled' if filled else 'unfilled'
    args = ['voxels', es, voxel_config.voxel_id, model_id, fs]
    return get_inference_subdir(*args)


def _get_base_voxel_dataset(
        model_id, edge_length_threshold=0.1, voxel_config=None, filled=False,
        auto_save=True):
    kwargs = dict(
        model_id=model_id,
        edge_length_threshold=edge_length_threshold,
        voxel_config=voxel_config,
        filled=filled
    )
    subdir = get_voxel_subdir(**kwargs)
    if auto_save:
        create_voxel_data(overwrite=False, **kwargs)

    return bvd.BinvoxDataset(subdir, mode='r')


def _flatten_dataset(dataset):

    def key_map_fn(args):
        folder = os.path.join(*args[:-1])
        if not os.path.isdir(folder):
            os.makedirs(folder)
        return os.path.join(folder, args[-1])

    def inverse_key_map_fn(subpath):
        return tuple(k for k in subpath.split('/') if len(k) > 0)

    return dataset.map_keys(
        key_map_fn, inverse_key_map_fn)


def get_voxel_dataset(
        model_id, edge_length_threshold=0.1, voxel_config=None, filled=False,
        auto_save=True):
    base_dataset = _get_base_voxel_dataset(
        model_id, edge_length_threshold=edge_length_threshold,
        voxel_config=voxel_config, filled=filled,
        auto_save=auto_save)

    dataset = _flatten_dataset(base_dataset)
    return dataset


def _create_unfilled_voxel_data(
        model_id, edge_length_threshold=0.1, voxel_config=None,
        overwrite=False):
    import numpy as np
    from progress.bar import IncrementalBar
    if voxel_config is None:
        voxel_config = _default_config
    mesh_dataset = get_inferred_mesh_dataset(
        model_id, edge_length_threshold)
    voxel_dataset = _get_base_voxel_dataset(
        model_id, edge_length_threshold, voxel_config, filled=False,
        auto_save=False)

    kwargs = dict(
        voxel_dim=voxel_config.voxel_dim,
        exact=voxel_config.exact,
        dc=voxel_config.dc,
        aw=voxel_config.aw)

    with mesh_dataset:
        print('Creating unfilled voxel data')
        for k, v in kwargs.items():
            print('%s = %s' % (k, v))
        bar = IncrementalBar(max=len(mesh_dataset))
        for k in mesh_dataset.keys():
            binvox_path = voxel_dataset.path(os.path.join(*k))
            if not os.path.isfile(binvox_path):
                mesh = mesh_dataset[k]
                vertices, faces = (
                    np.array(mesh[k]) for k in ('vertices', 'faces'))
                folder = os.path.dirname(binvox_path)
                if not os.path.isdir(folder):
                    os.makedirs(folder)
                bio.mesh_to_binvox(
                    vertices, faces, binvox_path, **kwargs)
            bar.next()
        bar.finish()


def _create_filled_voxel_data(**kwargs):
    from template_ffd.data.voxels import create_filled_data

    overwrite = kwargs.pop('overwrite', False)
    src = _get_base_voxel_dataset(filled=False, **kwargs)
    dst = bvd.BinvoxDataset(
        get_voxel_subdir(filled=True, **kwargs), mode='a')

    src = _flatten_dataset(src)
    dst = _flatten_dataset(dst)
    with src:
        with dst:
            message = 'Creating filled voxels'
            create_filled_data(
                src, dst, message=message, overwrite=overwrite)


def create_voxel_data(
        model_id, edge_length_threshold=0.1, voxel_config=None, filled=False,
        overwrite=False):
    kwargs = dict(
        model_id=model_id,
        edge_length_threshold=edge_length_threshold,
        voxel_config=voxel_config,
        overwrite=overwrite
    )
    if filled:
        _create_filled_voxel_data(**kwargs)
    else:
        _create_unfilled_voxel_data(**kwargs)
