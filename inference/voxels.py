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
        example_ids=None, auto_save=True):
    kwargs = dict(
        model_id=model_id,
        edge_length_threshold=edge_length_threshold,
        voxel_config=voxel_config,
        filled=filled
    )
    subdir = get_voxel_subdir(**kwargs)
    if auto_save:
        create_voxel_data(example_ids=example_ids, overwrite=False, **kwargs)

    return bvd.BinvoxDataset(subdir, mode='r')


def _flatten_dataset(dataset):
    def key_map_fn(*args):
        return os.path.join(*args)

    def inverse_key_map_fn(subpath):
        return tuple(k for k in subpath.split('/') if len(k) > 0)

    return dataset.map_keys(
        key_map_fn, inverse_key_map_fn)


def get_voxel_dataset(
        model_id, edge_length_threshold=0.1, voxel_config=None, filled=False,
        example_ids=None, auto_save=True):
    base_dataset = _get_base_voxel_dataset(
        model_id, edge_length_threshold=edge_length_threshold,
        voxel_config=voxel_config, filled=filled, example_ids=example_ids,
        auto_save=auto_save)

    return _flatten_dataset(base_dataset)


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
        bar = IncrementalBar(max=len(mesh_dataset))
        for k, mesh in mesh_dataset.items():
            bar.next()
            vertices, faces = (
                np.array(mesh[k]) for k in ('vertices', 'faces'))
            binvox_path = voxel_dataset.path(os.path.join(*k))
            # x, z, y = vertices.T
            # vertices = np.stack([x, y, z], axis=1)
            bio.mesh_to_binvox(
                vertices, faces, binvox_path, **kwargs)
        bar.finish()


def _create_filled_voxel_data(**kwargs):
    from template_ffd.data.voxels import create_filled_data

    overwrite = kwargs.pop('overwrite', False)
    src = _get_base_voxel_dataset(filled=False, **kwargs)
    kwargs.pop('example_ids')
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
        overwrite=False, example_ids=None):
    kwargs = dict(
        model_id=model_id,
        edge_length_threshold=edge_length_threshold,
        voxel_config=voxel_config,
        overwrite=overwrite,
        example_ids=example_ids
    )
    if filled:
        _create_filled_voxel_data(**kwargs)
    else:
        _create_unfilled_voxel_data(**kwargs)
