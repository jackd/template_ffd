#!/usr/bin/python


def create_split_mesh(
        cat_desc, edge_length_threshold, overwrite=False,
        start_threshold=None):
    """Create split mesh data for templates."""
    from shapenet.core import cat_desc_to_id
    from shapenet.core.meshes.config import get_mesh_config
    from template_ffd.templates.ids import get_template_ids
    cat_id = cat_desc_to_id(cat_desc)
    example_ids = get_template_ids(cat_id)
    config = get_mesh_config(edge_length_threshold)
    init = None if start_threshold is None else get_mesh_config(
        start_threshold)
    config.create_cat_data(cat_id, example_ids, overwrite, init)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('cat', type=str)
    parser.add_argument('edge_length_threshold', type=float)
    parser.add_argument(
        '-i', '--initial_edge_length_threshold', type=float)
    parser.add_argument('-o', '--overwrite', action='store_true')

    args = parser.parse_args()
    create_split_mesh(
        args.cat, args.edge_length_threshold, args.overwrite,
        args.initial_edge_length_threshold)
