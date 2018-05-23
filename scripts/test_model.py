#!/usr/bin/python


def main(model_id):
    import tensorflow as tf
    import tf_toolbox.testing
    from template_ffd.model import get_builder

    builder = get_builder(model_id)

    def get_train_op():
        features, labels = builder.get_train_inputs()
        return builder.get_estimator_spec(
            features, labels, tf.estimator.ModeKeys.TRAIN).train_op

    update_ops_run = tf_toolbox.testing.do_update_ops_run(get_train_op)
    tf_toolbox.testing.report_train_val_changes(get_train_op)

    if update_ops_run:
        print('Update ops run :)')
    else:
        print('Update ops not run :(')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', help='id of model defined in params')
    args = parser.parse_args()

    main(args.model_id)
