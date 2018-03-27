

def vis_clouds(
        model_id, pre_sampled=True, n_samples=1024, edge_length_threshold=0.1,
        shuffle=False):
    import random
    import numpy as np
    from mayavi import mlab
    import matplotlib.pyplot as plt
    from dids import Dataset
    from shapenet.core.blender_renderings.config import RenderConfig
    from shapenet.core.meshes import get_mesh_dataset
    from util3d.mayavi_vis import vis_point_cloud
    from util3d.mayavi_vis import vis_mesh
    from template_ffd.data.ids import get_example_ids
    from template_ffd.inference.clouds import get_inferred_cloud_dataset
    from template_ffd.model import get_builder
    builder = get_builder(model_id)
    cat_id = builder.cat_id
    kwargs = dict(model_id=model_id, n_samples=n_samples)
    if not pre_sampled:
        kwargs['edge_length_threshold'] = edge_length_threshold
    cloud_dataset = get_inferred_cloud_dataset(
        pre_sampled=pre_sampled, **kwargs)
    image_dataset = RenderConfig().get_dataset(cat_id, builder.view_index)

    example_ids = get_example_ids(cat_id, 'eval')
    if shuffle:
        example_ids = list(example_ids)
        random.shuffle(example_ids)
    mesh_dataset = get_mesh_dataset(cat_id)
    zipped_dataset = Dataset.zip(image_dataset, cloud_dataset, mesh_dataset)
    # zipped_dataset = Dataset.zip(image_dataset, cloud_dataset)
    with zipped_dataset:
        for example_id in example_ids:
            image, cloud, mesh = zipped_dataset[example_id]
            # image, cloud = zipped_dataset[example_id]
            plt.imshow(image)
            vis_point_cloud(
                np.array(cloud), color=(0, 1, 0), scale_factor=0.01)
            v, f = (np.array(mesh[k]) for k in ('vertices', 'faces'))
            vis_mesh(
                v, f, color=(0, 0, 1), opacity=0.1, include_wireframe=False)
            plt.show(block=False)
            mlab.show()
            plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_id', help='id of model defined in params')
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('-pre', '--pre_sampled', action='store_true')
    parser.add_argument('-n', '--n_samples', type=int, default=1024)
    parser.add_argument(
        '-t', '--edge_length_threshold', type=float, default=0.1)
    parser.add_argument('-s', '--shuffle', action='store_true')
    args = parser.parse_args()
    vis_clouds(
        args.model_id,
        args.pre_sampled,
        args.n_samples,
        args.edge_length_threshold,
        args.shuffle
    )
