"""Code for retrofitting code designed for single views to multiple views."""
import numpy as np
from template_ffd.model import get_builder


def retrofit_eval_fn(original_fn):
    def f(model_id, *args, **kwargs):
        if 'view_index' in kwargs:
            view_index = kwargs['view_index']
            if isinstance(view_index, int):
                return original_fn(model_id, *args, **kwargs)
            else:
                del kwargs['view_index']
        else:
            view_index = None
        if view_index is None:
            view_index = get_builder(model_id).view_index
        if isinstance(view_index, int):
            return original_fn(
                model_id, *args, view_index=view_index, **kwargs)
        assert(isinstance(view_index, (list, tuple)))
        values = [original_fn(model_id, *args, view_index=vi, **kwargs)
                  for vi in view_index]
        return np.mean(values)
    return f
