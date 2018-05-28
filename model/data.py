import numpy as np


def get_image_dataset(cat_ids, example_ids, view_indices, render_config=None):
    from shapenet.image import with_background
    from dids.core import BiKeyDataset
    if render_config is None:
        from shapenet.core.blender_renderings.config import RenderConfig
        render_config = RenderConfig()
    if isinstance(cat_ids, str):
        cat_ids = [cat_ids]
        example_ids = [example_ids]
    if isinstance(view_indices, int):
        view_indices = [view_indices]
    datasets = {
        c: render_config.get_multi_view_dataset(
            c, view_indices=view_indices, example_ids=eid)
        for c, eid in zip(cat_ids, example_ids)}
    dataset = BiKeyDataset(datasets).map(
        lambda image: with_background(image, 255))
    dataset = dataset.map_keys(
        lambda key: (key[0], (key[1], key[2])),
        lambda key: (key[0],) + key[1])
    return dataset


def get_cloud_dataset(cat_ids, example_ids, n_samples=16384, n_resamples=1024):
    import os
    from shapenet.core.point_clouds import PointCloudAutoSavingManager
    from util3d.point_cloud import sample_points
    from dids.core import BiKeyDataset
    if isinstance(cat_ids, str):
        cat_ids = [cat_ids]
        example_ids = [example_ids]
    datasets = {}
    for cat_id, e_ids in zip(cat_ids, example_ids):
        manager = PointCloudAutoSavingManager(cat_id, n_samples)
        if not os.path.isfile(manager.path):
            manager.save_all()
        datasets[cat_id] = manager.get_saving_dataset(
            mode='r').subset(e_ids)
    return BiKeyDataset(datasets).map(
        lambda x: sample_points(np.array(x, dtype=np.float32), n_resamples))


if __name__ == '__main__':
    from shapenet.core import cat_desc_to_id
    from template_ffd.data.ids import get_example_ids
    import random
    cat_ids = [cat_desc_to_id(i) for i in ('plane', 'car')]
    view_indices = [1, 5, 6]
    mode = 'train'
    example_ids = [get_example_ids(cat_id, mode) for cat_id in cat_ids]
    image_dataset = get_image_dataset(cat_ids, example_ids, view_indices)
    cloud_dataset = get_cloud_dataset(cat_ids, example_ids)

    image_dataset.open()
    cloud_dataset.open()

    keys = list((tuple(k) for k in image_dataset.keys()))
    random.shuffle(keys)

    def vis(image, cloud):
        import matplotlib.pyplot as plt
        from util3d.mayavi_vis import vis_point_cloud, mlab
        plt.imshow(image)
        plt.show(block=False)
        vis_point_cloud(
            cloud, axis_order='xzy', color=(0, 0, 1), scale_factor=0.02)
        mlab.show()
        plt.close()

    cat_ids, example_ids, view_indices = zip(*keys)
    for (cat_id, example_id, view_index) in zip(
            cat_ids, example_ids, view_indices):
        image = image_dataset[cat_id, example_id, view_index]
        cloud = cloud_dataset[cat_id, example_id]
        vis(image, cloud)
