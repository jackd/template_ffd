#!/usr/bin/python


def create_and_report(
        model_id, edge_length_threshold, filled, overwrite=False):
    import template_ffd.eval.iou as iou
    print(iou.get_iou_average(
        model_id=model_id,
        edge_length_threshold=edge_length_threshold,
        filled=filled))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_id', help='id of model defined in params')
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument(
        '-t', '--edge_length_threshold', type=float, default=0.02)
    # parser.add_argument('-f', '--filled', action='store_true')
    parser.add_argument('-ho', '--hollow', action='store_true')
    args = parser.parse_args()

    create_and_report(
        args.model_id,
        args.edge_length_threshold,
        not args.hollow,
        args.overwrite)
