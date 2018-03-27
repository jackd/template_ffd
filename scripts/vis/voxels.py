def vis_voxels(model_id, edge_length_threshold, filled, shuffle=False):
    from mayavi import mlab
    from util3d.mayavi_vis import vis_voxels
    from shapenet.core import cat_desc_to_id
    from template_ffd.inference.voxels import get_voxel_dataset
    from template_ffd.data.voxels import get_gt_voxel_dataset
    from template_ffd.model import load_params
    from template_ffd.data.ids import get_example_ids
    cat_id = cat_desc_to_id(load_params(model_id)['cat_desc'])
    gt_ds = get_gt_voxel_dataset(cat_id, filled)
    inf_ds = get_voxel_dataset(model_id, edge_length_threshold)
    example_ids = get_example_ids(cat_id, 'eval')
    if shuffle:
        example_ids = list(example_ids)
        example_ids.shuffle

    with gt_ds:
        with inf_ds:
            for example_id in example_ids:
                gt = gt_ds[example_id].data
                inf = inf_ds[example_id].data
                vis_voxels(gt, color=(0, 0, 1))
                mlab.figure()
                vis_voxels(inf, color=(0, 1, 0))
                mlab.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_id', help='id of model defined in params')
    parser.add_argument(
        '-t', '--edge_length_threshold', type=float, default=0.1)
    parser.add_argument('-s', '--shuffle', action='store_true')
    parser.add_argument('-f', '--filled', action='store_true')
    args = parser.parse_args()
    vis_voxels(
        args.model_id, args.edge_length_threshold, args.filled, args.shuffle)
