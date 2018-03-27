

def create_ffd(
        n=3, cat_descs=None, edge_length_threshold=None, n_samples=None,
        overwrite=False):
    from shapenet.core import cat_desc_to_id
    from template_ffd.templates.ids import get_templated_cat_ids
    from template_ffd.templates.ffd import create_ffd_data
    if cat_descs is None or len(cat_descs) == 0:
        cat_ids = get_templated_cat_ids()
    else:
        cat_ids = [cat_desc_to_id(c) for c in cat_descs]
    for cat_id in cat_ids:
        create_ffd_data(
            cat_id, n=n, edge_length_threshold=edge_length_threshold,
            n_samples=n_samples, overwrite=overwrite)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cats', default=None, nargs='*')
    parser.add_argument('-n', type=int, default=3)
    parser.add_argument(
        '-e', '--edge_length_threshold', default=None, type=float)
    parser.add_argument('-s', '--n_samples', default=None, type=int)
    parser.add_argument('-o', '--overwrite', action='store_true')

    args = parser.parse_args()
    create_ffd(
        args.n, args.cats, args.edge_length_threshold, args.n_samples,
        args.overwrite)
