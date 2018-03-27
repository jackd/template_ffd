
def main(model_id, mode):
    from template_ffd.model import get_builder
    builder = get_builder(model_id)
    builder.vis_inputs()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', help='id of model defined in params')
    parser.add_argument(
        '-m', '--mode', default='train', choices=['train', 'eval', 'infer'])
    args = parser.parse_args()

    main(args.model_id, args.mode)
