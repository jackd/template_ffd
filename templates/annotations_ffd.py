import dids.file_io.hdf5 as h
from ids import get_template_ids


def _calculate_ffd(n, vertices, points):
    import template_ffd.ffd.deform as ffd
    stu_origin, stu_axes = ffd.get_stu_params(vertices)
    dims = (n,) * 3
    return ffd.get_ffd(points, dims)


class FfdAnnotations(h.Hdf5AutoSavingManager):
    def __init__(self, cat_id, n=3):
        self._cat_id = cat_id
        self._n = n

    @property
    def saving_message(self):
        return (
            'Creating annotations FFD data\n'
            'cat_id: %s\n'
            'n: %d\n' % (self._cat_id, self._n))

    @property
    def path(self):
        import os
        from path import templates_dir
        return os.path.join(
            templates_dir, '_ffd', str(self._n), 'annotations',
            '%s.hdf5' % self._cat_id)

    def get_lazy_dataset(self):
        import numpy as np
        from shapenet.core.meshes import get_mesh_dataset
        from shapenet.core.annotations.datasets import PointCloudDataset
        from dids import Dataset
        vertices_dataset = get_mesh_dataset(self._cat_id).map(
            lambda mesh: np.array(mesh['vertices']))
        points_dataset = PointCloudDataset(self._cat_id)
        zipped = Dataset.zip(vertices_dataset, points_dataset)

        def map_fn(inputs):
            vertices, points = inputs
            b, p = _calculate_ffd(self._n, vertices, points)
            return dict(b=b, p=p)

        with points_dataset:
            keys = [k for k in get_template_ids(self._cat_id)
                    if k in points_dataset]

        return zipped.map(map_fn).subset(keys)


def get_annotations_ffd_dataset(cat_id, n=3):
    return FfdAnnotations(cat_id, n).get_saved_dataset()
