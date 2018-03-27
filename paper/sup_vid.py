import os
import random
from progress.bar import IncrementalBar
import numpy as np
from mayavi import mlab
from dids import Dataset
from template_ffd.inference.predictions import get_predictions_dataset
from template_ffd.model import get_builder
from template_ffd.templates.ffd import get_ffd_dataset
from template_ffd.templates.mesh import get_template_mesh_dataset
from shapenet.core.meshes import get_mesh_dataset
from shapenet.core.blender_renderings.config import RenderConfig


def get_source(vertices, faces, opacity=0.2):
    x, z, y = vertices.T
    mesh = mlab.triangular_mesh(
        x, y, z, faces, color=(0, 0, 1), opacity=opacity)
    return mesh.mlab_source


def update(source, vertices, angle_change=2):
    az, el, dist, focal = mlab.view()
    x, z, y = vertices.T
    source.set(x=x, y=y, z=z)
    mlab.view(az+angle_change, el, dist, focal)


def get_vertices(b, p, dp, n_frames):
    return [np.matmul(b, p + t*dp) for t in np.linspace(0, 1, n_frames)]


def vis_anim(b, p, dp, faces, duration, fps):
    n_frames = duration * fps
    delay = 1000 // fps
    angle_change = 360 // n_frames
    mlab.figure()
    vertices = get_vertices(b, p, dp, n_frames)
    source = get_source(vertices[0], faces)

    @mlab.animate(delay=delay)
    def anim():
        for v in vertices:
            update(source, v, angle_change=angle_change)
            yield

    anim()


def vis(b, p, dp, faces, gt_mesh, image, duration=5, fps=2):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    mlab.figure()
    v, f = (np.array(gt_mesh[k]) for k in ('vertices', 'faces'))
    x, z, y = v.T
    mlab.triangular_mesh(x, y, z, f, color=(0, 0, 1), opacity=0.2)
    mlab.figure()
    vis_anim(b, p, dp, faces, duration, fps)
    plt.show(block=False)
    mlab.show()
    plt.close()


def frame_fn(frame_index):
    return 'frame%04d.png' % frame_index


def save_frames(source, vertices, images_dir):
    print('Saving frames...')
    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)
    bar = IncrementalBar(max=len(vertices))
    angle_change = 360 // len(vertices)
    for i, v in enumerate(vertices):
        update(source, v, angle_change=angle_change)
        mlab.savefig(filename=os.path.join(images_dir, frame_fn(i)))
        bar.next()
    bar.finish()
    mlab.close()


def merge_frames(video_path, images_dir, fps=50):
    import subprocess
    subprocess.call([
        'ffmpeg',
        '-framerate', str(fps),
        '-i', '/%s/frame%%04d.png' % images_dir,
        '-c:v', 'libx264',
        '-profile:v', 'high',
        '-crf', '20',
        '-pix_fmt', 'yuv420p',
        video_path
    ])


def save_anim(subdir, b, p, dp, faces, duration=5, fps=50):
    n_frames = int(fps * duration)
    images_dir = os.path.join(subdir, 'video_frames')
    vertices = get_vertices(b, p, dp, n_frames)
    source = get_source(vertices[0], faces)
    save_frames(source, vertices, images_dir)
    video_path = os.path.join(subdir, 'deformation.mp4')
    merge_frames(video_path, images_dir, fps)


def save(subdir, b, p, dp, faces, gt_mesh, image, duration=5, fps=50):
    from scipy.misc import imsave
    from util3d.mesh.obj_io import write_obj
    imsave(os.path.join(subdir, 'image.png'), image)
    v, f = (np.array(gt_mesh[k]) for k in ('vertices', 'faces'))
    write_obj(os.path.join(subdir, 'gt_mesh.obj'), v, f)
    save_anim(subdir, b, p, dp, faces, duration, fps)


def get_data(model_id, example_ids=None):
    edge_length_threshold = 0.02
    builder = get_builder(model_id)
    cat_id = builder.cat_id

    with get_ffd_dataset(cat_id, edge_length_threshold=0.02) as ffd_ds:
        template_ids, bs, ps = zip(*builder.get_ffd_data(ffd_ds))

    with get_template_mesh_dataset(cat_id, edge_length_threshold) as mesh_ds:
        faces = [np.array(mesh_ds[e]['faces']) for e in template_ids]

    predictions_ds = get_predictions_dataset(model_id)
    mesh_ds = get_mesh_dataset(cat_id)
    image_ds = RenderConfig().get_dataset(cat_id, builder.view_index)
    zipped = Dataset.zip(predictions_ds, mesh_ds, image_ds)
    with zipped:
        if example_ids is None:
            example_ids = list(predictions_ds.keys())
            random.shuffle(example_ids)
        for example_id in example_ids:
            print(example_id)
            pred, mesh, image = zipped[example_id]
            i = np.argmax(pred['probs'])
            dp = np.array(pred['dp'][i])
            b = bs[i]
            p = ps[i]
            yield example_id, b, p, dp, faces[i], mesh, image


def vis_all(model_id, example_ids=None):
    for example_id, b, p, dp, f, mesh, image in get_data(
            model_id, example_ids):
        vis(b, p, dp, f, mesh, image)


def save_all(model_id, example_ids):
    root_dir = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), 'sup_vid_results',
        model_id)
    for example_id, b, p, dp, f, mesh, image in get_data(
            model_id, example_ids):
        subdir = os.path.join(root_dir, example_id)
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        save(subdir, b, p, dp, f, mesh, image, fps=50)


# cat_desc = 'chair'
# example_ids = [
#     '52cfbd8c8650402ba72559fc4f86f700',
#     # '8590bac753fbcccb203a669367e5b2a',
#     '353bbd3b916426d24502f857a1cf320e',
# ]

# cat_desc = 'plane'
# example_ids = [
#     '7bc46908d079551eed02ab0379740cae',
#     '5aeb583ee6e0e4ea42d0e83abdfab1fd',
#     'bbd8e6b06d8906d5eccd82bb51193a7f',
# ]

cat_desc = 'car'
example_ids = [
    # '7d7ace3866016bf6fef78228d4881428',
    '8d26c4ebd58fbe678ba7af9f04c27920',
    '764f08cd895e492e5dca6305fb9f97ca',
    'e2722a39dbc33044bbecf72e56fe7e5d'
]

# cat_desc = 'sofa'
# example_ids = [
#     '2e5d49e60a1f3abae9deec47d8412ee',
#     'db8c451f7b01ae88f91663a74ccd2338',
#     'e3b28c9216617a638ab9d2d7b1d714',
# ]


# cat_desc = 'table'
# example_ids = [
#     'd3fd6d332e6e8bccd5382f3f8f33a9f4',
#     '5d00596375ec8bd89940e75c3dc3e7',
#     # 'df7761a3b4ac638c9eaceb124b71b7be',
#     # '60ef2830979fd08ec72d4ae978770752',
#     # '5ac1ba406888f05e855931d119219022',
# ]

regime = 'e'
model_id = '%s_%s' % (regime, cat_desc)
# vis_all(model_id)
save_all(model_id, example_ids)
