def vis_mesh(model_id, edge_length_threshold, shuffle=False, wireframe=False):
    import numpy as np
    from mayavi import mlab
    from shapenet.core.meshes import get_mesh_dataset
    from shapenet.core import cat_desc_to_id
    from util3d.mayavi_vis import vis_mesh
    from template_ffd.inference.meshes import get_inferred_mesh_dataset
    from template_ffd.model import load_params
    import random

    def vis(mesh, **kwargs):
        v, f = (np.array(mesh[k]) for k in ('vertices', 'faces'))
        vis_mesh(v, f, include_wireframe=wireframe, **kwargs)

    cat_id = cat_desc_to_id(load_params(model_id)['cat_desc'])
    inf_mesh_dataset = get_inferred_mesh_dataset(
        model_id, edge_length_threshold)
    with inf_mesh_dataset:
        with get_mesh_dataset(cat_id) as gt_mesh_dataset:
            example_ids = list(inf_mesh_dataset.keys())
            if shuffle:
                random.shuffle(example_ids)

            for example_id in example_ids:
                inf = inf_mesh_dataset[example_id]
                gt = gt_mesh_dataset[example_id]
                mlab.figure()
                vis(inf, color=(0, 1, 0), opacity=0.2)
                mlab.figure()
                vis(gt, opacity=0.2)
                mlab.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_id', help='id of model defined in params')
    parser.add_argument('-t', '--edge_length_threshold', default=None,
                        type=float)
    parser.add_argument('-w', '--wireframe', action='store_true')
    parser.add_argument('-s', '--shuffle', action='store_true')
    args = parser.parse_args()
    vis_mesh(
        args.model_id, args.edge_length_threshold, args.wireframe,
        args.shuffle)
