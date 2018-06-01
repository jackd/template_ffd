import numpy as np
import dids.file_io.hdf5 as h
from path import get_split_mesh_group_path
from ids import get_template_ids


class SplitTemplateMeshManager(h.Hdf5AutoSavingManager):
    def __init__(self, cat_id, edge_length_threshold, initial_thresh=None):
        self._cat_id = cat_id
        self._edge_length_threshold = edge_length_threshold
        self._initial_thresh = initial_thresh
        if initial_thresh is not None and \
                initial_thresh <= edge_length_threshold:
            raise ValueError(
                'initial_thresh must be greater than edge_length_threshold')

    @property
    def path(self):
        return get_split_mesh_group_path(
            self._edge_length_threshold, self._cat_id)

    @property
    def saving_message(self):
        return ('Creating split template mesh data\n'
                'cat_id: %s\n'
                'edge_length_threshold: %s\n' %
                (self._cat_id, self._edge_length_threshold))

    def get_lazy_dataset(self):
        from util3d.mesh.edge_splitter import split_to_threshold
        base = get_template_mesh_dataset(self._cat_id, self._initial_thresh)
        base = base.subset(get_template_ids(self._cat_id))

        def map_fn(mesh):
            vertices, faces = (
                np.array(mesh[k]) for k in ('vertices', 'faces'))
            vertices, faces = split_to_threshold(
                vertices, faces, self._edge_length_threshold)
            return dict(vertices=np.array(vertices), faces=np.array(faces))

        return base.map(map_fn)


def get_split_template_mesh_dataset(cat_id, edge_length_threshold):
    return SplitTemplateMeshManager(
        cat_id, edge_length_threshold).get_saved_dataset()


def _get_template_mesh_dataset(cat_id, edge_length_threshold=None):
    import shapenet.core.meshes as m
    if edge_length_threshold is None:
        return m.get_mesh_dataset(cat_id).subset(get_template_ids(cat_id))
    else:
        return get_split_template_mesh_dataset(
            cat_id=cat_id,
            edge_length_threshold=edge_length_threshold)


def get_template_mesh_dataset(cat_id, edge_length_threshold=None):
    if isinstance(cat_id, (list, tuple)):
        from dids.core import BiKeyDataset
        datasets = {c: _get_template_mesh_dataset(c, edge_length_threshold)
                    for c in cat_id}
        return BiKeyDataset(datasets)
    else:
        return _get_template_mesh_dataset(cat_id, edge_length_threshold)
