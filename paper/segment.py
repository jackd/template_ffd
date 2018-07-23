#!/usr/bin/python

import os
import numpy as np
from mayavi import mlab
from util3d.mayavi_vis import vis_mesh
from template_ffd.inference.predictions import get_predictions_dataset
from template_ffd.model import get_builder
from template_ffd.data.ids import get_example_ids
from shapenet.core.blender_renderings.config import RenderConfig
import matplotlib.pyplot as plt
# from model.template import segment_faces

_paper_dir = os.path.realpath(os.path.dirname(__file__))

colors = [
    # (1, 1, 1),
    (0, 0, 1),
    (0, 1, 0),
    (1, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
    (0, 0, 0),
]
nc = len(colors)


def segmented_cloud(cloud, segmentation):
    assert(np.min(segmentation) >= 1)
    for i in range(1, np.max(segmentation)+1):
        yield cloud[segmentation == i]


def vis_clouds(clouds):
    for i, cloud in enumerate(clouds):
        x, z, y = cloud.T
        mlab.points3d(x, y, z, color=colors[i % nc], scale_factor=0.01)


def vis_segmented_mesh(vertices, faces, **kwargs):
    for i, f in enumerate(faces):
        vis_mesh(vertices, f, color=colors[i % nc], **kwargs)


def vis_segmentations(
        model_id, example_ids=None, vis_mesh=False,
        edge_length_threshold=0.02, include_wireframe=False,
        save=False):
    from scipy.misc import imsave
    if save and example_ids is None:
        raise ValueError('Cannot save without specifying example_ids')
    builder = get_builder(model_id)
    cat_id = builder.cat_id
    if example_ids is None:
        example_ids = example_ids = get_example_ids(cat_id, 'eval')
    if vis_mesh:
        segmented_fn = builder.get_segmented_mesh_fn(edge_length_threshold)
    else:
        segmented_fn = builder.get_segmented_cloud_fn()
    config = RenderConfig()

    with get_predictions_dataset(model_id) as predictions:
        with config.get_dataset(cat_id, builder.view_index) as image_ds:
            for example_id in example_ids:
                example = predictions[example_id]
                probs, dp = (np.array(example[k]) for k in ('probs', 'dp'))
                result = segmented_fn(probs, dp)
                if result is not None:
                    image = image_ds[example_id]
                    print(example_id)
                    segmentation = result['segmentation']
                    if vis_mesh:
                        vertices = result['vertices']
                        faces = result['faces']
                        original_points = result['original_points']
                        original_seg = result['original_segmentation']
                        f0 = mlab.figure(bgcolor=(1, 1, 1))
                        vis_segmented_mesh(
                            vertices, segmented_cloud(faces, segmentation),
                            include_wireframe=include_wireframe,
                            opacity=0.2)
                        f1 = mlab.figure(bgcolor=(1, 1, 1))
                        vis_clouds(
                            segmented_cloud(original_points, original_seg))
                    else:
                        points = result['points']
                        original_points = result['original_points']
                        f0 = mlab.figure(bgcolor=(1, 1, 1))
                        vis_clouds(segmented_cloud(points, segmentation))
                        f1 = mlab.figure(bgcolor=(1, 1, 1))
                        vis_clouds(
                            segmented_cloud(original_points, segmentation))

                    if save:
                        folder = os.path.join(
                            _paper_dir, 'segmentations', model_id, example_id)
                        if not os.path.isdir(folder):
                            os.makedirs(folder)
                        fn = 'inferred_%s.png' % (
                            'mesh' if vis_mesh else 'cloud')
                        p0 = os.path.join(folder, fn)
                        mlab.savefig(p0, figure=f0)
                        p1 = os.path.join(folder, 'annotated_cloud.png')
                        mlab.savefig(p1, figure=f1)
                        pi = os.path.join(folder, 'query_image.png')
                        imsave(pi, image)
                        mlab.close()
                    else:
                        plt.imshow(image)
                        plt.show(block=False)
                        mlab.show()
                        plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_id', help='id of model defined in params')
    parser.add_argument('-i', '--example_ids', type=str, nargs='*')
    parser.add_argument('-m', '--mesh', action='store_true')
    parser.add_argument('-t', '--edge_length_threshold', default=0.02)
    parser.add_argument('-w', '--wireframe', action='store_true')
    parser.add_argument('-s', '--save', action='store_true')
    args = parser.parse_args()
    model_id = args.model_id
    example_ids = args.example_ids
    if isinstance(example_ids, (list, tuple)) and len(example_ids) == 0:
        example_ids = None

    vis_segmentations(
        model_id, example_ids, args.mesh, args.edge_length_threshold,
        args.wireframe, args.save)
