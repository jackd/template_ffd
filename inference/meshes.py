from dids.file_io.hdf5 import Hdf5AutoSavingManager
from template_ffd.model import get_builder
from path import get_inference_path


class InferredMeshManager(Hdf5AutoSavingManager):
    def __init__(self, model_id, edge_length_threshold=0.1):
        self._model_id = model_id
        self._edge_length_threshold = edge_length_threshold
        self._nested_depth = 3

    @property
    def path(self):
        elt = self._edge_length_threshold
        es = 'base' if elt is None else str(elt)
        return get_inference_path('meshes', es, '%s.hdf5' % self._model_id)

    @property
    def saving_message(self):
        return (
            'Saving mesh data\n'
            'model_id: %s\n'
            'edge_length_threshold: %s\n' %
            (self._model_id, self._edge_length_threshold))

    def get_lazy_dataset(self):
        from predictions import get_predictions_dataset
        builder = get_builder(self._model_id)
        mesh_fn = builder.get_prediction_to_mesh_fn(
            self._edge_length_threshold)

        def map_fn(prediction):
            mesh = mesh_fn(**prediction)
            return {k: mesh[k] for k in ('vertices', 'faces', 'attrs')}

        mesh_ds = get_predictions_dataset(self._model_id).map(map_fn)
        return mesh_ds


def get_inferred_mesh_dataset(
        model_id, edge_length_threshold=0.1, lazy=True):
    manager = InferredMeshManager(model_id, edge_length_threshold)
    if lazy:
        return manager.get_lazy_dataset()
    else:
        return manager.get_saved_dataset()
