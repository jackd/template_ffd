import numpy as np
from util3d.mesh.sample import sample_faces
from dids.file_io.hdf5 import Hdf5AutoSavingManager
from path import get_inference_path
from template_ffd.model import get_builder


class _PostSampledCloudManager(Hdf5AutoSavingManager):
    def __init__(self, model_id, n_samples=1024, edge_length_threshold=0.1):
        self._model_id = model_id
        self._n_samples = n_samples
        self._edge_length_threshold = edge_length_threshold

    @property
    def path(self):
        return get_inference_path(
            'cloud', 'postsampled',
            str(self._n_samples), str(self._edge_length_threshold),
            '%s.hdf5' % self._model_id)

    @property
    def saving_message(self):
        return (
            'Generating postampled point cloud\n'
            'model_id: %s\n'
            'n_samples: %d\n'
            'edge_length_threshold: %s' % (
                self._model_id, self._n_samples, self._edge_length_threshold))

    def get_lazy_dataset(self):
        from meshes import get_inferred_mesh_dataset

        def map_fn(mesh):
            vertices, faces = (
                np.array(mesh[k]) for k in ('vertices', 'faces'))
            cloud = sample_faces(vertices, faces, self._n_samples)
            return cloud

        return get_inferred_mesh_dataset(
            self._model_id, self._edge_length_threshold).map(map_fn)


class _PreSampledCloudManager(Hdf5AutoSavingManager):
    def __init__(self, model_id, n_samples=1024):
        self._model_id = model_id
        self._n_samples = n_samples

    @property
    def path(self):
        return get_inference_path(
            'cloud', 'presampled', str(self._n_samples),
            '%s.hdf5' % self._model_id)

    @property
    def saving_message(self):
        return (
            'Generating presampled point cloud\n'
            'model_id: %s\n'
            'n_samples: %d' % (self._model_id, self._n_samples))

    def get_lazy_dataset(self):
        from predictions import get_predictions_dataset
        builder = get_builder(self._model_id)
        cloud_fn = builder.get_prediction_to_cloud_fn(self._n_samples)

        def map_fn(predictions):
            return cloud_fn(**predictions)['cloud']

        return get_predictions_dataset(self._model_id).map(map_fn)


def get_cloud_manager(model_id, pre_sampled=False, **kwargs):
    if pre_sampled:
        return _PreSampledCloudManager(model_id, **kwargs)
    else:
        return _PostSampledCloudManager(model_id, **kwargs)


def get_inferred_cloud_dataset(model_id, pre_sampled=False, **kwargs):
    return get_cloud_manager(
        model_id, pre_sampled, **kwargs).get_saved_dataset()
