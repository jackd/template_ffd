#!/usr/bin/python


def create_and_report(
        pre_sampled, model_id, n_samples, cat_desc, edge_length_threshold,
        overwrite=False):
    import template_ffd.eval.chamfer as chamfer
    kwargs = dict(
        pre_sampled=pre_sampled,
        model_id=model_id,
        n_samples=n_samples,
        cat_desc=cat_desc,
        edge_length_threshold=edge_length_threshold,
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
    parser.add_argument('-c', '--cat_desc', type=str, nargs='*')
    parser.add_argument(
        '-t', '--edge_length_threshold', type=float, default=0.02)
    args = parser.parse_args()

    create_and_report(
        not args.post_sampled,
        args.model_id,
        args.n_samples,
        args.cat_desc,
        args.edge_length_threshold,
        args.overwrite)
