import numpy as np
from dids import Dataset
from shapenet.core.point_clouds import get_point_cloud_dataset
from util3d.point_cloud import sample_points
from normalize import get_normalization_params_dataset, normalized
from template_ffd.data.ids import get_example_ids


def get_lazy_evaluation_dataset(inf_cloud_ds, cat_id, n_samples, eval_fn):
    example_ids = get_example_ids(cat_id, 'eval')

    def sample_fn(cloud):
        return sample_points(np.array(cloud), n_samples)

    normalization_ds = get_normalization_params_dataset(cat_id)
    gt_cloud_ds = get_point_cloud_dataset(
        cat_id, n_samples, example_ids=example_ids).map(sample_fn)
    zipped = Dataset.zip(inf_cloud_ds, gt_cloud_ds, normalization_ds)

    def map_fn(data):
        inf_cloud, gt_cloud, norm_params = data
        inf_cloud = normalized(inf_cloud, **norm_params)
        gt_cloud = normalized(gt_cloud, **norm_params)
        return eval_fn(inf_cloud, gt_cloud)

    return zipped.map(map_fn)
