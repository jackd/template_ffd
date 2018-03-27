import os

templates_dir = os.path.realpath(os.path.dirname(__file__))
template_ids_path = os.path.join(templates_dir, 'templates.json')


def get_ffd_group_dir(n=3, edge_length_threshold=None, n_samples=None):
    root = os.path.join(templates_dir, '_ffd', str(n))
    if n_samples is None:
        es = 'base' if edge_length_threshold is None else \
            str(edge_length_threshold)
        return os.path.join(root, 'mesh', es)
    else:
        if edge_length_threshold is not None:
            raise ValueError(
                'Cannot have both n_samples and edge_length_threshold')
        return os.path.join(root, 'sampled', str(n_samples))


def get_ffd_group_path(
        cat_id, n=3, edge_length_threshold=None, n_samples=None):
    return os.path.join(get_ffd_group_dir(n, edge_length_threshold, n_samples),
                        '%s.hdf5' % cat_id)


def get_split_mesh_group_dir(edge_length_threshold):
    d = os.path.join(templates_dir, '_split_mesh', str(edge_length_threshold))
    if not os.path.isdir(d):
        os.makedirs(d)
    return d


def get_split_mesh_group_path(edge_length_threshold, cat_id):
    return os.path.join(
        get_split_mesh_group_dir(edge_length_threshold), '%s.hdf5' % cat_id)
