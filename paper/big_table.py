#!/usr/bin/python

import random
import numpy as np
from dids import Dataset
from shapenet.core.blender_renderings.config import RenderConfig
from shapenet.core.meshes import get_mesh_dataset
from shapenet.core import cat_desc_to_id

from template_ffd.inference.clouds import get_cloud_manager
from template_ffd.inference.meshes import get_inferred_mesh_dataset
from template_ffd.inference.voxels import get_voxel_dataset
from template_ffd.inference.predictions import \
    get_selected_template_idx_dataset
from template_ffd.data.ids import get_example_ids
from template_ffd.templates.ids import get_template_ids


def get_ds(cat_desc, regime='e'):
    view_index = 5
    edge_length_threshold = 0.02

    cat_id = cat_desc_to_id(cat_desc)
    model_id = '%s_%s' % (regime, cat_desc)

    image_ds = RenderConfig().get_dataset(cat_id, view_index)
    cloud_ds = get_cloud_manager(
        model_id, pre_sampled=True, n_samples=n_samples).get_lazy_dataset()
    mesh_ds = get_inferred_mesh_dataset(
        model_id, edge_length_threshold=edge_length_threshold)
    gt_mesh_ds = get_mesh_dataset(cat_id)
    voxel_ds = get_voxel_dataset(
        model_id, edge_length_threshold=edge_length_threshold, filled=False)
    selected_template_ds = get_selected_template_idx_dataset(model_id)

    template_meshes = []
    with gt_mesh_ds:
        for template_id in get_template_ids(cat_id):
            mesh = gt_mesh_ds[template_id]
            template_meshes.append(
                {k: np.array(mesh[k]) for k in ('vertices', 'faces')})

    template_mesh_ds = selected_template_ds.map(lambda i: template_meshes[i])

    return Dataset.zip(
        image_ds, gt_mesh_ds, cloud_ds, mesh_ds, voxel_ds, template_mesh_ds)


def vis(cat_desc, regime='e', shuffle=True):
    import matplotlib.pyplot as plt
    from mayavi import mlab
    from util3d.mayavi_vis import vis_point_cloud, vis_voxels

    all_ds = get_ds(cat_desc, regime)
    cat_id = cat_desc_to_id(cat_desc)
    example_ids = list(get_example_ids(cat_id, 'eval'))
    random.shuffle(example_ids)

    def vis_mesh(mesh, include_wireframe=False, **kwargs):
        from util3d.mayavi_vis import vis_mesh as vm
        v, f = (np.array(mesh[k]) for k in ('vertices', 'faces'))
        vm(v, f, include_wireframe=include_wireframe, **kwargs)

    with all_ds:
        for example_id in example_ids:
            print(example_id)
            image, gt_mesh, cloud, mesh, voxels, template_mesh = \
                all_ds[example_id]
            plt.imshow(image)
            mlab.figure()
            vis_mesh(gt_mesh, color=(0, 0, 1))
            mlab.figure()
            vis_mesh(mesh, color=(0, 1, 0))
            mlab.figure()
            vis_mesh(template_mesh, color=(1, 0, 0))
            mlab.figure()
            vis_point_cloud(
                np.array(cloud), scale_factor=0.01, color=(0, 1, 0))
            mlab.figure()
            vis_voxels(voxels.data, color=(0, 1, 0))

            plt.show(block=False)
            mlab.show()
            plt.close()


def export(cat_desc, example_ids, regime='e'):
    from scipy.misc import imsave
    from util3d.mesh.obj_io import write_obj
    import os
    all_ds = get_ds(cat_desc, regime)
    base = os.path.realpath(os.path.dirname(__file__))

    with all_ds:
        for example_id in example_ids:
            folder = os.path.join(
                base, 'big_table_results', cat_desc, example_id)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            image, gt_mesh, cloud, mesh, voxels, template_mesh = \
                all_ds[example_id]
            imsave(os.path.join(folder, 'image.png'), image)
            v, f = (np.array(mesh[k]) for k in ('vertices', 'faces'))
            write_obj(os.path.join(folder, 'deformed.obj'), v, f)
            v, f = (np.array(template_mesh[k]) for k in ('vertices', 'faces'))
            write_obj(os.path.join(folder, 'template.obj'), v, f)
            v, f = (np.array(gt_mesh[k]) for k in ('vertices', 'faces'))
            write_obj(os.path.join(folder, 'model.obj'), v, f)
            np.save(os.path.join(
                folder, 'inferred_cloud.npy'), np.array(cloud))
            path = os.path.join(folder, 'deformed.binvox')
            voxels.save(path)


n_samples = 8192
regime = 'e'

# cat_desc = 'chair'
# example_ids = [
#     '52cfbd8c8650402ba72559fc4f86f700',
#     # '8590bac753fbcccb203a669367e5b2a',
#     '353bbd3b916426d24502f857a1cf320e',
# ]

# cat_desc = 'plane'
# example_ids = [
#     '7bc46908d079551eed02ab0379740cae',
#     '5aeb583ee6e0e4ea42d0e83abdfab1fd',
#     'bbd8e6b06d8906d5eccd82bb51193a7f',
# ]

# cat_desc = 'car'
# example_ids = [
#     # '7d7ace3866016bf6fef78228d4881428',
#     '8d26c4ebd58fbe678ba7af9f04c27920',
#     '764f08cd895e492e5dca6305fb9f97ca',
#     'e2722a39dbc33044bbecf72e56fe7e5d'
# ]

# cat_desc = 'sofa'
# example_ids = [
#     '2e5d49e60a1f3abae9deec47d8412ee',
#     'db8c451f7b01ae88f91663a74ccd2338',
#     'e3b28c9216617a638ab9d2d7b1d714',
# ]

cat_desc = 'table'
example_ids = [
    'd3fd6d332e6e8bccd5382f3f8f33a9f4',
    '5d00596375ec8bd89940e75c3dc3e7',
    '5ac1ba406888f05e855931d119219022',
]

# vis(cat_desc, regime)
export(cat_desc, example_ids, regime)
