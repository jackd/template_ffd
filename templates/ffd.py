import os
import numpy as np
import dids.file_io.hdf5 as h
from path import get_ffd_group_path
from mesh import get_template_mesh_dataset


def _calculate_ffd(vertices, faces, n=3, n_samples=None):
    import template_ffd.ffd.deform as ffd
    import util3d.mesh.sample as sample
    stu_origin, stu_axes = ffd.get_stu_params(vertices)
    if n_samples is None:
        points = vertices
    else:
        points = sample.sample_faces(vertices, faces, n_samples)
    dims = (n,) * 3
    return ffd.get_ffd(points, dims)


class FfdManager(h.Hdf5AutoSavingManager):
    def __init__(
            self, cat_id, n=3, edge_length_threshold=None, n_samples=None):
        self._cat_id = cat_id
        self._n = n
        self._edge_length_threshold = edge_length_threshold
        self._n_samples = n_samples

    @property
    def saving_message(self):
        return (
            'Creating FFD data\n'
            'cat_id: %s\n'
            'n: %d\n'
            'edge_length_threshold: %s\n'
            'n_samples: %s' % (
                self._cat_id, self._n, self._edge_length_threshold,
                self._n_samples))

    @property
    def path(self):
        return get_ffd_group_path(
            self._cat_id,
            self._n,
            self._edge_length_threshold,
            self._n_samples)

    def get_lazy_dataset(self):
        base = get_template_mesh_dataset(
            self._cat_id, self._edge_length_threshold)

        def map_fn(base):
            vertices, faces = (
                np.array(base[k]) for k in ('vertices', 'faces'))
            b, p = _calculate_ffd(vertices, faces, self._n, self._n_samples)
            return dict(b=b, p=p)

        return base.map(map_fn)


def create_ffd_data(
        cat_id, n=3, edge_length_threshold=None, n_samples=None,
        overwrite=False):
    FfdManager(cat_id, n, edge_length_threshold, n_samples).save_all(
        overwrite=overwrite)


def get_ffd_dataset(cat_id, n=3, edge_length_threshold=None, n_samples=None):
    manager = FfdManager(
        cat_id=cat_id,
        n=n,
        edge_length_threshold=edge_length_threshold,
        n_samples=n_samples)
    if not os.path.isfile(manager.path):
        return manager.get_saved_dataset()
    else:
        return manager.get_saving_dataset()
