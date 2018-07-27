#!/usr/bin/python


def create_inferred_meshes(model_id, edge_length_threshold):
    from template_ffd.inference.meshes import get_inferred_mesh_dataset
    get_inferred_mesh_dataset(model_id, edge_length_threshold, lazy=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_id', help='id of model defined in params')
    # parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument(
        '-t', '--edge_length_threshold', type=float, default=0.02)
    args = parser.parse_args()

    create_inferred_meshes(
        args.model_id,
        args.edge_length_threshold,
        # args.overwrite
        )
