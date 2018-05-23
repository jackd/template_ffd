#!/usr/bin/python


def create_voxels(
        model_id, edge_length_threshold, filled, overwrite, cat_desc):
    if model_id is None:
        if cat_desc is None:
            raise ValueError('One of model_id or cat_desc must be supplied')
        from template_ffd.data.voxels import create_filled_gt_data
        from shapenet.core import cat_desc_to_id
        cat_id = cat_desc_to_id(cat_desc)
        create_filled_gt_data(cat_id, overwrite=overwrite)
    else:
        from template_ffd.inference.voxels import create_voxel_data
        create_voxel_data(
            model_id, edge_length_threshold, filled=filled,
            overwrite=overwrite)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', type=str, default=None, nargs='?')
    parser.add_argument('-c', '--cat', type=str, default=None)
    parser.add_argument(
        '-t', '--edge_length_threshold', type=float, default=0.1)
    parser.add_argument('-f', '--filled', action='store_true')
    parser.add_argument('-o', '--overwrite', action='store_true')

    args = parser.parse_args()
    create_voxels(
        args.model_id, args.edge_length_threshold, args.filled, args.overwrite,
        cat_desc=args.cat)
