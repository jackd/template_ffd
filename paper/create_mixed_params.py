"""Creates params file for single model trained across 13 categories."""
import os
import json
from progress.spinner import Spinner
from template_ffd.model import load_params, get_params_path, get_builder
# from shapenet.core import cat_desc_to_id

path = get_params_path('e_all_v8')
if os.path.isfile(path):
    os.remove(path)
    # print('Path %s already exists.')
    # exit()


def get_template_counts(model_id):
    import tensorflow as tf
    import numpy as np
    print('Getting template counts for %s' % model_id)
    graph = tf.Graph()
    with graph.as_default():
        builder = get_builder(model_id)
        features, labels = builder.get_inputs(mode='train', repeat=False)
        spec = builder.get_estimator_spec(features, labels, mode='eval')
        predictions = spec.predictions
        probs = predictions['probs']
        counts = tf.argmax(probs, axis=-1)
        totals = np.zeros((builder.n_templates,), dtype=np.int32)
        saver = tf.train.Saver()

        with tf.train.MonitoredSession() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(builder.model_dir))
            spinner = Spinner()
            while not sess.should_stop():
                c = sess.run(counts)
                for ci in c:
                    totals[ci] += 1
                spinner.next()
                # break
            spinner.finish()
    return totals


def get_top_k(x, k):
    print(x)
    ret = x.argsort()[-k:][::-1]
    print('---')
    print(ret)
    return list(ret)


descs = (
    'plane',
    'bench',
    'car',
    'chair',
    'sofa',
    'cabinet',
    'monitor',
    'lamp',
    'speaker',
    'pistol',
    'table',
    'cellphone',
    'watercraft',
)

model_ids = tuple('e_%s_v8' % c for c in descs)
template_idx = tuple(get_top_k(get_template_counts(m), 2) for m in model_ids)
params = load_params(model_ids[0])

params['cat_desc'] = descs
params['template_idxs'] = template_idx
params['use_bn_bugged_version'] = False

with open(path, 'w') as fp:
    json.dump(params, fp)
