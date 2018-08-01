import numpy as np
from dids.core import Dataset
# from dids.core import BiKeyDataset
from shapenet.core.point_clouds import get_point_cloud_dataset
from util3d.point_cloud import sample_points
from normalize import get_normalization_params_dataset, normalized
from template_ffd.data.ids import get_example_ids


def _get_lazy_evaluation_dataset_single(
        inf_cloud_ds, cat_id, n_samples, eval_fn):

    def sample_fn(cloud):
        return sample_points(np.array(cloud), n_samples)

    example_ids = get_example_ids(cat_id, 'eval')

    normalization_ds = get_normalization_params_dataset(cat_id)
    gt_cloud_ds = get_point_cloud_dataset(
        cat_id, n_samples, example_ids=example_ids).map(sample_fn)

    with inf_cloud_ds:
        keys = tuple(inf_cloud_ds.keys())

    normalization_ds = normalization_ds.map_keys(
        lambda key: key[:2])
    gt_cloud_ds = gt_cloud_ds.map_keys(lambda key: key[:2])

    zipped = Dataset.zip(
        inf_cloud_ds, gt_cloud_ds, normalization_ds).subset(
            keys, check_present=False)

    def map_fn(data):
        inf_cloud, gt_cloud, norm_params = data
        inf_cloud = normalized(inf_cloud, **norm_params)
        gt_cloud = normalized(gt_cloud, **norm_params)
        return eval_fn(inf_cloud, gt_cloud)

    dataset = zipped.map(map_fn)
    return dataset


def get_lazy_evaluation_dataset(inf_cloud_ds, cat_id, n_samples, eval_fn):
    if not isinstance(cat_id, (list, tuple)):
        cat_id = [cat_id]
    return _get_lazy_evaluation_dataset_single(
        inf_cloud_ds, cat_id, n_samples, eval_fn)

    # if isinstance(cat_id, (list, tuple)):
    #     return _get_lazy_evaluation_dataset_single(
    #             inf_cloud_ds, cat_id, n_samples, eval_fn)

    # def f(cid):
    #     return _get_lazy_evaluation_dataset_single(
    #         inf_cloud_ds, cid, n_samples, eval_fn)
    # if isinstance(cat_id, (list, tuple)):
    #     datasets = {k: f(k) for k in cat_id}
    #     return BiKeyDataset(datasets)
    # else:
    #     return f(cat_id)
