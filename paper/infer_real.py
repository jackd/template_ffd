import os

_paper_dir = os.path.realpath(os.path.dirname(__file__))


def get_path(cat_id, example_id, ext='png'):
    return os.path.join(
        _paper_dir, 'real_images', 'final', cat_id,
        '%s.%s' % (example_id, ext))


def vis_mesh(vertices, faces, original_vertices, **kwargs):
    from util3d.mayavi_vis import vis_mesh
    from mayavi import mlab
    mlab.figure()
    vis_mesh(
        vertices=vertices, faces=faces, color=(0, 1, 0),
        include_wireframe=False)
    mlab.figure()
    vis_mesh(
        vertices=original_vertices, faces=faces, color=(1, 0, 0),
        include_wireframe=False)
    mlab.show()


def get_inference(model_id, example_id, ext='png', edge_length_threshold=0.02):
    import tensorflow as tf
    from template_ffd.model import get_builder
    import PIL
    import numpy as np
    from shapenet.image import with_background
    builder = get_builder(model_id)
    cat_id = builder.cat_id

    example_ids = [example_id]
    paths = [get_path(cat_id, e, ext) for e in example_ids]
    for path in paths:
        if not os.path.isfile(path):
            raise Exception('No file at path %s' % path)

    def gen():
        for example_id, path in zip(example_ids, paths):
            image = np.array(PIL.Image.open(path))
            image = with_background(image, 255)
            yield example_id, image

    render_params = builder.params.get('render_params', {})
    shape = tuple(render_params.get('shape', (192, 256)))
    shape = shape + (3,)

    def input_fn():
        ds = tf.data.Dataset.from_generator(
            gen, (tf.string, tf.uint8), ((), shape))
        example_id, image = ds.make_one_shot_iterator().get_next()
        # image_content = tf.read_file(path)
        # if ext == 'png':
        #     image = tf.image.decode_png(image_content)
        # elif ext == 'jpg':
        #     image = tf.image.decode_jpg(image_content)
        # else:
        #     raise ValueError('ext must be in ("png", "jpg")')
        image.set_shape((192, 256, 3))
        image = tf.image.per_image_standardization(image)
        example_id = tf.expand_dims(example_id, axis=0)
        image = tf.expand_dims(image, axis=0)
        return dict(example_id=example_id, image=image)

    estimator = builder.get_estimator()
    mesh_fn = builder.get_prediction_to_mesh_fn(edge_length_threshold)
    for pred in estimator.predict(input_fn):
        example_id = pred.pop('example_id')
        mesh = mesh_fn(**pred)
        vis_mesh(**mesh)


if __name__ == '__main__':
    model_id = 'b_plane'
    example_id = 'bomber-00'
    get_inference(model_id, example_id, 'png')
