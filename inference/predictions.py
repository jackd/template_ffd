from dids.file_io.hdf5 import Hdf5Dataset
from shapenet.util import LengthedGenerator
from path import get_inference_path
from template_ffd.model import get_builder
from template_ffd.data.ids import get_example_ids


def get_predictions_data_path(model_id):
    return get_inference_path('predictions', '%s.hdf5' % model_id)


def get_predictions_data(model_id):
    builder = get_builder(model_id)
    cat_id = builder.cat_id
    mode = 'infer'
    example_ids = get_example_ids(cat_id, mode)

    estimator = builder.get_estimator()
    predictions = estimator.predict(builder.get_predict_inputs)
    return LengthedGenerator(predictions, len(example_ids))


def create_predictions_data(model_id, overwrite=False):

    def map_fn(prediction):
        example_id, probs, dp = (
            prediction[k] for k in ('example_id', 'probs', 'dp'))
        return example_id, dict(probs=probs, dp=dp)

    predictions = get_predictions_data(model_id)
    mapped = (map_fn(p) for p in predictions)
    gen = LengthedGenerator(mapped, len(predictions))

    with _get_predictions_dataset(model_id, mode='a') as dataset:
        dataset.save_items(gen, overwrite=overwrite)


def _get_predictions_dataset(model_id, mode):
    return Hdf5Dataset(get_predictions_data_path(model_id), mode)


def get_predictions_dataset(model_id, mode='r'):
    import os
    path = get_predictions_data_path(model_id)
    if not os.path.isfile(path):
        print('Creating predictions data')
        create_predictions_data(model_id)
    return _get_predictions_dataset(model_id, mode)


def get_selected_template_idx_dataset(model_id):
    import numpy as np

    def map_fn(pred):
        return np.argmax(np.array(pred['probs']))

    return get_predictions_dataset(model_id).map(map_fn)
