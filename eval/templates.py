import numpy as np
from template_ffd.inference.predictions import get_predictions_dataset
from template_ffd.model import get_builder


def print_template_scores(model_id, by_weight=False):
    builder = get_builder(model_id)
    template_ids = builder.template_ids
    n = len(template_ids)
    counts = np.zeros((n,), dtype=np.int32)
    totals = np.zeros((n,), dtype=np.float32)
    dataset = get_predictions_dataset(model_id)

    with dataset:
        for example_id in dataset:
            probs = np.array(dataset[example_id]['probs'])
            counts[np.argmax(probs)] += 1
            totals += probs

    if by_weight:
        zipped = list(zip(template_ids, range(n), totals))
        zipped.sort(key=lambda x: x[2], reverse=True)
        for rank, (k, i, p) in enumerate(zipped):
            print(rank, i, p, k)
        print([z[1] for z in zipped])
    else:
        zipped = list(zip(template_ids, range(n), counts))
        zipped.sort(key=lambda x: x[2], reverse=True)
        for rank, (k, i, p) in enumerate(zipped):
            print(rank, i, p, k)
        print([z[1] for z in zipped])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_id', help='id of model defined in params')
    parser.add_argument('-w', '--by_weight', action='store_true')
    args = parser.parse_args()
    print_template_scores(args.model_id, args.by_weight)
