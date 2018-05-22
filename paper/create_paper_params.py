import os
import json
from template_ffd.model import get_params_path


def write_params(model_id, params):
    path = get_params_path(model_id)
    if not os.path.isfile(path):
        with open(path, 'w') as fp:
            json.dump(params, fp)
        print('Wrote params for %s' % model_id)


cats = (
    'plane', 'car', 'bench', 'chair', 'sofa',  'table', 'cabinet', 'monitor',
    'lamp', 'speaker', 'watercraft', 'cellphone', 'pistol')
param_types = ('b', 'e', 'w', 'r')

params = {
    'b': {},
    'e': {
        'entropy_loss': {
                'weight': 1e2,
                'exp_annealing_rate': 1e-4
        }
    },
    'w': {
        'gamma': 'log',
        'prob_eps': 1e-3
    },
    'r': {
        'dp_regularization': {
            'weight': 1e0,
            'exp_annealing_rate': 1e-4
        }
    },
    'rm1': {
        'dp_regularization': {
            'weight': 1e-1,
            'exp_annealing_rate': 1e-4
        }
    }
}
for k, v in params.items():
    v['inference_params'] = {'alpha': 0.25}

for cat in cats:
    for p in param_types:
        ps = params[p]
        ps['cat_desc'] = cat
        model_id = '%s_%s' % (p, cat)
        write_params(model_id, ps)

    # multi view param sets
    src = get_params_path('e_%s' % cat)
    model_id = 'e_%s_v8' % cat

    with open(src, 'r') as fp:
        ps = json.load(fp)

    ps['view_index'] = range(8)
    write_params(model_id, ps)


# # TODO: change template ids
# ps = params['e'].copy()
# ps['cat_desc'] = cats[:5]
# write_params('e_all-5', ps)
# ps['view_index'] = range(8)
# write_params('e_all-5_v8', ps)
# del ps['view_index']
# ps['cat_desc'] = cats
# write_params('e_all-13', ps)
# ps['view_index'] = range(8)
# write_params('e_all-13_v8', ps)
