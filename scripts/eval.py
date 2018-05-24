#!/usr/bin/python


def eval_model(model_id):
    import tensorflow as tf
    from template_ffd.model import get_builder
    tf.logging.set_verbosity(tf.logging.INFO)
    builder = get_builder(model_id)
    builder.initialize_variables()
    print(builder.eval())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_id', help='id of model defined in params')
    args = parser.parse_args()
    eval_model(args.model_id)
