import numpy as np
from dids.file_io.json_dataset import JsonAutoSavingManager
from template_ffd.data.ids import get_example_ids
from path import get_eval_path


def get_normalization_params(vertices):
    from scipy.optimize import minimize
    vertices = np.array(vertices)
    vertical_offset = np.min(vertices[:, 1])
    vertices[:, 1] -= vertical_offset

    def f(x):
        x = np.array([x[0], 0, x[1]])
        dist2 = np.sum((vertices - x)**2, axis=-1)
        return np.max(dist2)

    opt = minimize(f, np.array([0, 0])).x
    offset = np.array([opt[0], vertical_offset, opt[1]], dtype=np.float32)
    vertices[:, [0, 2]] -= opt

    radius = np.sqrt(np.max(np.sum(vertices**2, axis=-1)))
    unit1 = 3.2
    scale_factor = radius / unit1
    return offset, scale_factor


def normalized(points, offset, scale_factor):
    return (points - offset) / scale_factor


def normalize(points, offset, scale_factor):
    points -= offset
    points /= scale_factor


class _NormalizationParamsAutoSavingManager(JsonAutoSavingManager):
    def __init__(self, cat_id):
        self._cat_id = cat_id

    @property
    def saving_message(self):
        return (
            'Creating transform parameter dataset\ncat_id: %s' % self._cat_id)

    @property
    def path(self):
        return get_eval_path('transform_params', '%s.json' % self._cat_id)

    def get_lazy_dataset(self):
        from shapenet.core.meshes import get_mesh_dataset
        example_ids = get_example_ids(self._cat_id, 'eval')
        mesh_ds = get_mesh_dataset(self._cat_id).subset(example_ids)

        def map_fn(mesh):
            vertices = mesh['vertices']
            offset, scale_factor = get_normalization_params(vertices)
            return dict(
                offset=[float(o) for o in offset],
                scale_factor=float(scale_factor))

        return mesh_ds.map(map_fn)


def get_normalization_params_dataset(cat_id):
    dataset = _NormalizationParamsAutoSavingManager(cat_id).get_saved_dataset()
    return dataset.map(
        lambda x: {k: np.array(v, dtype=np.float32) for k, v in x.items()})
