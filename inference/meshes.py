from dids.file_io.hdf5 import Hdf5AutoSavingManager
from template_ffd.model import get_builder
from path import get_inference_path


class InferredMeshManager(Hdf5AutoSavingManager):
    def __init__(self, model_id, edge_length_threshold=0.1, view_index=None):
        self._model_id = model_id
        self._edge_length_threshold = edge_length_threshold
        self._view_index = view_index

    @property
    def path(self):
        elt = self._edge_length_threshold
        es = 'base' if elt is None else str(elt)
        args = ['meshes', es]
        view_index = self._view_index
        if view_index is None:
            args.append('%s.hdf5' % self._model_id)
        else:
            args.extend((self._model_id, 'v%d.hdf5' % view_index))
        return get_inference_path(*args)

    @property
    def saving_message(self):
        return (
            'Saving mesh data\n'
            'model_id: %s\n'
            'edge_length_threshold: %s\n'
            'view_index: %s' %
            (self._model_id, self._edge_length_threshold,
             str(self._view_index)))

    def get_lazy_dataset(self):
        from predictions import get_predictions_dataset
        builder = get_builder(self._model_id)
        mesh_fn = builder.get_prediction_to_mesh_fn(
            self._edge_length_threshold)

        def map_fn(prediction):
            mesh = mesh_fn(**prediction)
            return {k: mesh[k] for k in ('vertices', 'faces', 'attrs')}

        return get_predictions_dataset(
            self._model_id, self._view_index).map(map_fn)


def get_inferred_mesh_dataset(
        model_id, edge_length_threshold=0.1, view_index=None, lazy=True):
    manager = InferredMeshManager(
        model_id, edge_length_threshold, view_index)
    if lazy:
        return manager.get_lazy_dataset()
    else:
        return manager.get_saved_dataset()
