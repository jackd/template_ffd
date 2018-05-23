#!/usr/bin/python
"""Script for checking if a given model has predictions for all example_ids."""


def check_predictions(model_id):
    from template_ffd.inference.predictions import get_predictions_dataset
    from template_ffd.model import get_builder
    from template_ffd.data.ids import get_example_ids
    builder = get_builder(model_id)
    cat_id = builder.cat_id
    example_ids = get_example_ids(cat_id, 'eval')

    missing = []
    with get_predictions_dataset(model_id, 'r') as dataset:
        for example_id in example_ids:
            if example_id not in dataset:
                missing.append(example_id)
            else:
                example = dataset[example_id]
                if not all(k in example for k in ('probs', 'dp')):
                    missing.append(example_id)

    if len(missing) == 0:
        print('No predictions missing!')
    else:
        print('%d / %d predictions missing' % (len(missing), len(example_ids)))
        for example_id in example_ids:
            print(example_id)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id')
    args = parser.parse_args()
    check_predictions(args.model_id)
