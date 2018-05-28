#!/usr/bin/python


def generate_inferences(
        model_id, overwrite=False):
    from template_ffd.inference.predictions import create_predictions_data
    create_predictions_data(model_id, overwrite=overwrite)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_id', help='id of model defined in params')
    parser.add_argument('-v', '--view_index', default=None, type=int)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()
    generate_inferences(
        args.model_id, overwrite=args.overwrite)
