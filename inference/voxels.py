from path import get_inference_subdir
import util3d.voxel.dataset as bvd
import util3d.voxel.convert as bio
from shapenet.core.voxels.config import VoxelConfig
from meshes import get_inferred_mesh_dataset

_default_config = VoxelConfig()


def get_voxel_subdir(model_id, edge_length_threshold=0.1, voxel_config=None,
                     filled=False, view_index=None):
    if voxel_config is None:
        voxel_config = _default_config
    es = 'base' if edge_length_threshold is None else \
        str(edge_length_threshold)
    fs = 'filled' if filled else 'unfilled'
    args = ['voxels', es, voxel_config.voxel_id, model_id, fs]
    if view_index is not None:
        args.append('v%d' % view_index)
    return get_inference_subdir(*args)


def get_voxel_dataset(
        model_id, edge_length_threshold=0.1, voxel_config=None, filled=False,
        example_ids=None, view_index=None, auto_save=True):
    kwargs = dict(
        model_id=model_id,
        edge_length_threshold=edge_length_threshold,
        voxel_config=voxel_config,
        filled=filled,
        view_index=view_index
    )
    subdir = get_voxel_subdir(**kwargs)
    if auto_save:
        create_voxel_data(example_ids=example_ids, overwrite=False, **kwargs)
    return bvd.BinvoxDataset(subdir, mode='r')


def _create_unfilled_voxel_data(
        model_id, edge_length_threshold=0.1, voxel_config=None,
        view_index=None, overwrite=False, example_ids=None):
    from template_ffd.data.ids import get_example_ids
    from shapenet.core import cat_desc_to_id
    from template_ffd.model import load_params
    import numpy as np
    from progress.bar import IncrementalBar
    if voxel_config is None:
        voxel_config = _default_config
    cat_id = cat_desc_to_id(load_params(model_id)['cat_desc'])
    if example_ids is None:
        example_ids = get_example_ids(cat_id, 'eval')
    mesh_dataset = get_inferred_mesh_dataset(
        model_id, edge_length_threshold, view_index=view_index)
    voxel_dataset = get_voxel_dataset(
        model_id, edge_length_threshold, voxel_config, filled=False,
        auto_save=False, view_index=view_index)
    if not overwrite:
        example_ids = [i for i in example_ids if i not in voxel_dataset]
    if len(example_ids) == 0:
        return
    print('Creating %d voxels for model %s, view %s'
          % (len(example_ids), model_id, str(view_index)))

    kwargs = dict(
        voxel_dim=voxel_config.voxel_dim,
        exact=voxel_config.exact,
        dc=voxel_config.dc,
        aw=voxel_config.aw)

    with mesh_dataset:
        bar = IncrementalBar(max=len(example_ids))
        for example_id in example_ids:
            bar.next()
            mesh = mesh_dataset[example_id]
            vertices, faces = (
                np.array(mesh[k]) for k in ('vertices', 'faces'))
            binvox_path = voxel_dataset.path(example_id)
            # x, z, y = vertices.T
            # vertices = np.stack([x, y, z], axis=1)
            bio.mesh_to_binvox(
                vertices, faces, binvox_path, **kwargs)
        bar.finish()


def _create_filled_voxel_data(**kwargs):
    from template_ffd.data.voxels import create_filled_data

    overwrite = kwargs.pop('overwrite', False)
    unfilled = get_voxel_dataset(filled=False, **kwargs)
    kwargs.pop('example_ids')
    dst = bvd.BinvoxDataset(get_voxel_subdir(filled=True, **kwargs), mode='a')
    with unfilled:
        with dst:
            message = 'Creating filled voxels'
            create_filled_data(
                unfilled, dst, message=message, overwrite=overwrite)


def create_voxel_data(
        model_id, edge_length_threshold=0.1, voxel_config=None, filled=False,
        overwrite=False, example_ids=None, view_index=None):
    kwargs = dict(
        model_id=model_id,
        edge_length_threshold=edge_length_threshold,
        voxel_config=voxel_config,
        overwrite=overwrite,
        example_ids=example_ids,
        view_index=view_index
    )
    if filled:
        _create_filled_voxel_data(**kwargs)
    else:
        _create_unfilled_voxel_data(**kwargs)
