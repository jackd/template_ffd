
def train(model_id, max_steps):
    import tensorflow as tf
    from template_ffd.model import get_builder
    tf.logging.set_verbosity(tf.logging.INFO)
    builder = get_builder(model_id)
    builder.initialize_variables()
    builder.train(max_steps=max_steps)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_id', help='id of model defined in params')
    parser.add_argument('-s', '--max-steps', default=1e5, type=float)
    args = parser.parse_args()
    train(args.model_id, max_steps=args.max_steps)
