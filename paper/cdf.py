from __future__ import division


def analyse(cat_desc, templates=False, save=False, metric='chamfer'):
    if templates:
        n = [1, 2, 4, 8, 16, 30]
        model_ids = [('et%d_%s' % (i, cat_desc)) for i in n]
        model_ids[-1] = 'e_%s' % cat_desc
        labels = ['$T = %d$' % i for i in n]
        colors = [
            'orange',
            'g',
            'b',
            'k',
            'cyan',
            'r',
        ]
        fig_name = '%s_%s_T' % (cat_desc, metric)
        if metric == 'iou':
            legend_size = 10
        else:
            legend_size = 10
    else:
        labels = ['b', 'w', 'e', 'r']
        model_ids = ['%s_%s' % (m, cat_desc) for m in labels]
        colors = [
            'r',
            'orange',
            'g',
            'b',
        ]
        fig_name = '%s_%s' % (cat_desc, metric)
        legend_size = 10
    analyse_models(
        model_ids, labels, colors, save, metric, fig_name, title=None,
        legend_size=legend_size)


def analyse_models(model_ids, labels, colors, save=False, metric='chamfer',
                   fig_name=None, title=None, legend_size=None):
    import numpy as np
    import matplotlib.pyplot as plt

    template_vals = []
    model_vals = []

    if metric == 'chamfer':
        from template_ffd.eval.chamfer import get_chamfer_manager
        from template_ffd.eval.chamfer import get_template_chamfer_manager

        def template_fn(i):
            return get_template_chamfer_manager(i).get_saved_dataset()

        def model_fn(i):
            return get_chamfer_manager(i).get_saved_dataset()

        plot_fn = plt.semilogx

        def value_map_fn(x):
            return x

        reverse = False
        ylabel = '$\lambda_c < X$'
        xlim = None
        neg = False
        loc = 'lower right'
        # xlim = [2e-2, 1.1]

    elif metric == 'iou':
        from template_ffd.eval.iou import get_iou_dataset
        from template_ffd.eval.iou import IouTemplateSavingManager

        def template_fn(i):
            return IouTemplateSavingManager(i).get_saved_dataset()

        def model_fn(i):
            return get_iou_dataset(i, filled=True, edge_length_threshold=0.02)

        plot_fn = plt.plot
        # plot_fn = plt.semilogx

        def value_map_fn(x):
            # return [1 - xi for xi in x]
            return x

        reverse = True
        ylabel = '$IoU > X$'
        xlim = [0, 1]
        neg = False
        loc = 'upper right'

    else:
        raise ValueError('metric %s not recognized' % metric)

    xlabel = '$X$'

    for model_id in model_ids:
        with template_fn(model_id) as ds:
            values = value_map_fn(list(ds.values()))
        values.sort(reverse=reverse)
        template_vals.append(values)
        with model_fn(model_id) as ds:
            values = value_map_fn(list(ds.values()))
        values.sort(reverse=reverse)
        model_vals.append(values)

    fig = plt.figure()
    for mv, tv, label, i, c in zip(
            model_vals, template_vals, labels, model_ids, colors):
        n = len(mv)
        cdf = (np.array(range(n)) + 1) / n
        if neg:
            cdf = 1 - cdf
        plot_fn(mv, cdf, color=c, label=label, linestyle='dashed')
        plot_fn(tv, cdf, color=c, linestyle='dotted')
    ax = plt.gca()
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    if title is not None:
        plt.title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)

    # if legend_size is not None:
    #     plt.legend(prop={'size': legend_size})
    ax.legend(loc=loc)

    if save:
        import os
        folder = os.path.join(
            os.path.realpath(os.path.dirname(__file__)), 'figs')
        if not os.path.isdir(folder):
            os.makedirs(folder)
        fn = os.path.join(folder, '%s.eps' % fig_name)
        fig.savefig(fn, format='eps')
    else:
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'cat', help='cat_desc to analyse')
    parser.add_argument('-s', '--save', action='store_true')
    parser.add_argument('-t', '--templates', action='store_true')
    parser.add_argument(
        '-m', '--metric', default='chamfer', choices=['chamfer', 'iou'])
    args = parser.parse_args()
    analyse(args.cat, args.templates, args.save, args.metric)
