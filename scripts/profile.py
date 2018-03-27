
def main(model_id, skip_runs=10):
    import os
    import tensorflow as tf
    from template_ffd.model import get_builder
    from tf_toolbox.profile import create_profile
    builder = get_builder(model_id)

    def graph_fn():
        mode = tf.estimator.ModeKeys.TRAIN
        features, labels = builder.get_inputs(mode)
        spec = builder.get_estimator_spec(features, labels, mode)
        return spec.train_op

    folder = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), '_profiles')
    if not os.path.isdir(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, '%s.json' % model_id)

    create_profile(graph_fn, filename, skip_runs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', help='id of model defined in params')
    parser.add_argument('-s', '--skip_runs', type=int, default=10)
    args = parser.parse_args()

    main(args.model_id)
