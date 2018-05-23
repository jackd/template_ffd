#!/usr/bin/python


def create_and_report(
        pre_sampled, model_id, n_samples, edge_length_threshold,
        view_index, overwrite=False):
    import template_ffd.eval.chamfer as chamfer
    kwargs = dict(
        pre_sampled=pre_sampled,
        model_id=model_id,
        n_samples=n_samples,
        edge_length_threshold=edge_length_threshold,
        view_index=view_index
    )
    if pre_sampled:
        kwargs.pop('edge_length_threshold')
    mean = chamfer.get_chamfer_average(**kwargs)
    print(mean)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_id', help='id of model defined in params')
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('-post', '--post_sampled', action='store_true')
    parser.add_argument('-n', '--n_samples', type=int, default=1024)
    parser.add_argument(
        '-t', '--edge_length_threshold', type=float, default=0.02)
    parser.add_argument('-v', '--view_index', default=None, type=int)
    args = parser.parse_args()

    create_and_report(
        not args.post_sampled,
        args.model_id,
        args.n_samples,
        args.edge_length_threshold,
        args.view_index,
        args.overwrite)
