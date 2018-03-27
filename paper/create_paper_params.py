import os
import json
from template_ffd.model import get_params_path

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
        v = params[p]
        v['cat_desc'] = cat
        model_id = '%s_%s' % (p, cat)
        path = get_params_path(model_id)
        if not os.path.isfile(path):
            with open(path, 'w') as fp:
                json.dump(v, fp)
            print('Wrote params for %s' % model_id)
