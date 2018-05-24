from dids.file_io.hdf5 import Hdf5Dataset
from shapenet.util import LengthedGenerator
from path import get_inference_path
from template_ffd.model import get_builder
from template_ffd.data.ids import get_example_ids


def get_predictions_data_path(model_id, view_index=None):
    if view_index is None:
        return get_inference_path('predictions', '%s.hdf5' % model_id)
    else:
        return get_inference_path(
            'predictions', model_id, 'v%d.hdf5' % view_index)


def get_predictions_data(model_id, view_index=None):
    builder = get_builder(model_id)
    original_view_index = builder.view_index
    if view_index is None and not isinstance(original_view_index, int):
        raise ValueError('Must specify view_index if ')

    builder.params['view_index'] = view_index
    assert(builder.view_index == view_index)

    cat_id = builder.cat_id
    mode = 'infer'
    example_ids = get_example_ids(cat_id, mode)

    estimator = builder.get_estimator()
    predictions = estimator.predict(builder.get_predict_inputs)
    return LengthedGenerator(predictions, len(example_ids))


def _create_predictions_data(model_id, view_index, overwrite):
    def map_fn(prediction):
        example_id, probs, dp = (
            prediction[k] for k in ('example_id', 'probs', 'dp'))
        return example_id, dict(probs=probs, dp=dp)

    predictions = get_predictions_data(model_id, view_index)
    mapped = (map_fn(p) for p in predictions)
    gen = LengthedGenerator(mapped, len(predictions))

    with _get_predictions_dataset(model_id, view_index, mode='a') as dataset:
        dataset.save_items(gen, overwrite=overwrite)


def create_predictions_data(model_id, view_index=None, overwrite=False):
    if view_index is None:
        from template_ffd.model import load_params
        params = load_params(model_id)
        if 'view_index' in params:
            view_index = params['view_index']
            if isinstance(view_index, int):
                view_index = None
    if view_index is None:
        _create_predictions_data(model_id, view_index, overwrite)
    else:
        if isinstance(view_index, int):
            _create_predictions_data(model_id, view_index, overwrite)
        elif hasattr(view_index, '__iter__'):
            for vi in view_index:
                _create_predictions_data(model_id, vi, overwrite)
        else:
            raise TypeError('Invalivd view_index %s' % view_index)


def _get_predictions_dataset(model_id, view_index, mode):
    return Hdf5Dataset(get_predictions_data_path(model_id, view_index), mode)


def get_predictions_dataset(model_id, view_index=None, mode='r'):
    import os
    path = get_predictions_data_path(model_id, view_index)
    if not os.path.isfile(path):
        print('Creating predictions data')
        create_predictions_data(model_id, view_index)
    return _get_predictions_dataset(model_id, view_index, mode)


def get_selected_template_idx_dataset(model_id, view_index=None):
    import numpy as np

    def map_fn(pred):
        return np.argmax(np.array(pred['probs']))

    return _get_predictions_dataset(model_id, view_index).map(map_fn)
