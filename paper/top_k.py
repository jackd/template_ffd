#!/usr/bin/python

import random
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

from dids import Dataset
from shapenet.core.blender_renderings.config import RenderConfig
from shapenet.core.meshes import get_mesh_dataset
from shapenet.core import cat_desc_to_id

from template_ffd.inference.predictions import get_predictions_dataset
from template_ffd.data.ids import get_example_ids
from template_ffd.model import get_builder


regime = 'e'
cat_desc = 'chair'
view_index = 5
edge_length_threshold = 0.02

shuffle = True
k = 3

cat_id = cat_desc_to_id(cat_desc)
model_id = '%s_%s' % (regime, cat_desc)
builder = get_builder(model_id)

image_ds = RenderConfig().get_dataset(cat_id, view_index)
gt_mesh_ds = get_mesh_dataset(cat_id)
predictions_ds = get_predictions_dataset(model_id)

top_k_mesh_fn = builder.get_prediction_to_top_k_mesh_fn(
    edge_length_threshold, k)

all_ds = Dataset.zip(image_ds, gt_mesh_ds, predictions_ds)


def vis():

    def vis_mesh(mesh, include_wireframe=False, **kwargs):
        from util3d.mayavi_vis import vis_mesh as vm
        v, f = (np.array(mesh[k]) for k in ('vertices', 'faces'))
        vm(v, f, include_wireframe=include_wireframe, **kwargs)

    example_ids = list(get_example_ids(cat_id, 'eval'))
    random.shuffle(example_ids)

    with all_ds:
        for example_id in example_ids:
            print(example_id)
            image, gt_mesh, predictions = all_ds[example_id]
            meshes = top_k_mesh_fn(
                *(np.array(predictions[k]) for k in ('probs', 'dp')))
            plt.imshow(image)
            mlab.figure()
            vis_mesh(gt_mesh, color=(0, 0, 1))
            for mesh in meshes:
                v, f, ov = (mesh[k] for k in
                            ('vertices', 'faces', 'original_vertices'))
                mlab.figure()
                vis_mesh({'vertices': v, 'faces': f}, color=(0, 1, 0))
                mlab.figure()
                vis_mesh({'vertices': ov, 'faces': f}, color=(1, 0, 0))

            plt.show(block=False)
            mlab.show()
            plt.close()


def export(example_id):
    import os
    from util3d.mesh.obj_io import write_obj
    from scipy.misc import imsave
    save_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                            'top_k_results', example_id)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with all_ds:
        print(example_id)
        image, gt_mesh, predictions = all_ds[example_id]
        meshes = top_k_mesh_fn(
            *(np.array(predictions[k]) for k in ('probs', 'dp')))
        for i, mesh in enumerate(meshes):
            ov, v, f = (
                mesh[k] for k in ('original_vertices', 'vertices', 'faces'))
            write_obj(os.path.join(save_dir, 'template%d.obj' % i), ov, f)
            write_obj(os.path.join(save_dir, 'deformed%d.obj' % i), v, f)
        v, f = (np.array(gt_mesh[k]) for k in ('vertices', 'faces'))
        write_obj(os.path.join(save_dir, 'ground_truth.obj'), v, f)
        imsave(os.path.join(save_dir, 'image.png'), image)


# chair
export('114f72b38dcabdf0823f29d871e57676')
