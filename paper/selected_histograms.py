from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def get_hist_data(model_id, n_bins, mode):
    from shapenet.core import cat_desc_to_id
    from template_ffd.templates.ids import get_template_ids
    from template_ffd.model import load_params
    from template_ffd.inference.predictions import get_predictions_dataset
    cat_id = cat_desc_to_id(load_params(model_id)['cat_desc'])
    n_templates = len(get_template_ids(cat_id))

    counts = np.zeros((n_bins,), dtype=np.int32)
    argmax_counts = np.zeros((n_templates,), dtype=np.int32)

    with get_predictions_dataset(model_id) as dataset:
        for example_id in dataset:
            probs = np.array(dataset[example_id]['probs'])
            counts[int(np.max(probs) * n_bins)] += 1
            # prob_indices = np.array(0.999*probs * n_bins, dtype=np.int32)
            # for pi in prob_indices:
            #     counts[pi] += 1
            argmax_counts[np.argmax(probs)] += 1

    counts = counts / np.sum(counts)
    argmax_counts = argmax_counts / np.sum(argmax_counts)
    return counts, argmax_counts


def analyse(cat_desc, save):

    model_ids = [
        'b_',
        'w_',
        'e_',
        'r_',
    ]

    model_ids = ['%s%s' % (m, cat_desc) for m in model_ids]

    labels = [
        'b',
        'w',
        'e',
        'r',
    ]

    colors = [
        'r',
        'orange',
        'g',
        'b',
    ]

    mode = 'eval'
    n_bins = 10

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    n_ids = len(model_ids)

    counts = []
    argmax_counts = []

    for model_id in model_ids:
        c, ac = get_hist_data(model_id, n_bins, mode)
        counts.append(c)
        argmax_counts.append(ac)

    for i in range(n_ids):
        argmax_counts[i] = np.array(
            sorted(list(argmax_counts[i]), reverse=True))

    w = 0.9 / n_ids
    dx = 0.05 / n_ids + np.arange(n_ids)*w - 0.3

    n_templates = len(argmax_counts[0])
    n_templates = 15
    x = np.array(range(n_templates))
    template_fig = plt.figure()
    # plt.title(cat_desc)
    ax = plt.gca()
    for i in range(n_ids):
        ax.bar(x + dx[i], argmax_counts[i][:n_templates],
               width=w, color=colors[i], align='center', label=labels[i])
    plt.xlabel('Template', fontsize=20)
    plt.ylabel('Normalized frequency', fontsize=20)
    plt.xticks(np.arange(n_templates), ('',)*n_templates)

    ax.legend(loc='upper right')
    plt.legend(prop={'size': 20})

    x = np.array(range(n_bins)) / n_bins
    w = 0.9 / (n_ids*n_bins)
    offset = (1./n_bins - n_ids*w) / 2
    dx = np.array(range(n_ids)) * w + offset
    # gamma_fig = plt.figure()
    # ax = plt.gca()
    # for i in range(n_ids):
    #     ax.bar(x + dx[i], counts[i], width=w, color=colors[i],
    #            align='edge', label=labels[i])
    # plt.xlabel('$\max_t\gamma^{(t)}$')
    # plt.ylabel('Normalized frequency')
    # ax.legend(loc='upper right')

    if save:
        import os
        folder = os.path.join(
            os.path.realpath(os.path.dirname(__file__)), 'figs')
        if not os.path.isdir(folder):
            os.makedirs(folder)
        fn = os.path.join(folder, '%s_template_counts.eps' % cat_desc)
        template_fig.savefig(fn, format='eps')
        # gamma_fig.savefig('max_gamma_counts.eps', format='eps')
    else:
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'cat', help='cat_desc to analyse')
    parser.add_argument('-s', '--save', action='store_true')
    args = parser.parse_args()
    analyse(args.cat, args.save)
