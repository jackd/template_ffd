import os
import numpy as np
from scipy.misc import imread
cat_desc = 'car'
regime = 'r'
model_id = '%s_%s' % (regime, cat_desc)

folder = os.path.join(
    os.path.realpath(
        os.path.dirname(__file__)), 'real_images', 'jhonys', cat_desc)

fns = [fn for fn in os.listdir(folder) if fn[-4:] == '.png']


def save():
    import tensorflow as tf
    from util3d.mesh.obj_io import write_obj
    from shapenet.image import with_background
    from template_ffd.model import get_builder
    builder = get_builder(model_id)

    mesh_fn = builder.get_prediction_to_mesh_fn(0.02)
    cloud_fn = builder.get_prediction_to_cloud_fn()

    graph = tf.Graph()
    with graph.as_default():
        image = tf.placeholder(shape=(192, 256, 3), dtype=tf.uint8)
        std_image = tf.image.per_image_standardization(image)
        std_image = tf.expand_dims(std_image, axis=0)
        example_id = tf.constant(['blah'], dtype=tf.string)
        spec = builder.get_estimator_spec(
            dict(example_id=example_id, image=std_image),
            None, tf.estimator.ModeKeys.PREDICT)
        predictions = spec.predictions
        probs_tf = predictions['probs']
        dp_tf = predictions['dp']
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, tf.train.latest_checkpoint(builder.model_dir))
        for fn in fns:
            path = os.path.join(folder, fn)
            image_data = np.array(imread(path))
            if image_data.shape[-1] == 4:
                image_data = with_background(image_data, (255, 255, 255))
            probs, dp = sess.run(
                [probs_tf, dp_tf], feed_dict={image: image_data})
            probs = probs[0]
            dp = dp[0]
            mesh = mesh_fn(probs, dp)
            cloud = cloud_fn(probs, dp)['cloud']
            v, ov, f = (
                mesh[k] for k in('vertices', 'original_vertices', 'faces'))
            path = '%s.obj' % path[:-4]
            write_obj(path, v, f)
            p2 = '%s_template.obj' % path[:-4]
            np.save('%s_cloud.npy' % path[:-4], cloud)
            write_obj(p2, ov, f)


def vis():
    from util3d.mesh.obj_io import parse_obj
    from util3d.mayavi_vis import vis_mesh
    from mayavi import mlab
    import matplotlib.pyplot as plt

    for fn in fns:
        path = os.path.join(folder, fn)
        image = imread(path)
        p0 = '%s.obj' % path[:-4]
        vertices, faces = parse_obj(p0)[:2]
        p1 = '%s_template.obj' % path[:-4]
        tv, tf = parse_obj(p1)[:2]
        # cloud = np.load('%s_cloud.npy' % path[:-4])
        assert(np.all(tf == faces))
        print(np.max(np.abs(vertices - tv)))
        # mlab.figure()
        # vis_point_cloud(cloud, color=(0, 1, 0), scale_factor=0.02)
        mlab.figure()
        vis_mesh(vertices, faces, include_wireframe=False, color=(0, 1, 0))
        mlab.figure()
        vis_mesh(tv, tf, include_wireframe=False)
        plt.figure()
        plt.imshow(image)
        # plt.show(block=False)
        # mlab.show()
        # plt.close()
    plt.show(block=False)
    mlab.show()


save()
vis()
