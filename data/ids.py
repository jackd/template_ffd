import os

_ids_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), '_ids')


class SplitConfig(object):
    def __init__(self, train_prop=0.8, seed=0):
        self._train_prop = train_prop
        self._seed = seed
        self._split_config_id = 's%d-%s' % (seed, train_prop)
        root_dir = os.path.join(_ids_dir, self._split_config_id)
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        self._root_dir = root_dir

    @property
    def root_dir(self):
        return self._root_dir

    def get_txt_path(self, cat_id, mode):
        return os.path.join(self._root_dir, '%s_%s.txt' % (cat_id, mode))

    def _get_example_ids(self, cat_id, mode):
        if not self.has_split(cat_id):
            self.create_split(cat_id, overwrite=True)
        if mode in ('predict', 'infer'):
            mode = 'eval'
        with open(self.get_txt_path(cat_id, mode)) as fp:
            example_ids = [i.rstrip() for i in fp.readlines()]
        return example_ids

    def get_example_ids(self, cat_id, mode):
        if isinstance(cat_id, (list, tuple)):
            return tuple(self._get_example_ids(c, mode) for c in cat_id)
        else:
            return self._get_example_ids(cat_id, mode)

    def has_split(self, cat_id):
        return all(os.path.isfile(self.get_txt_path(cat_id, m))
                   for m in ('train', 'eval'))

    def create_split(self, cat_id, overwrite=False):
        import random
        from shapenet.core import get_example_ids
        from template_ffd.templates.ids import get_template_ids
        if not overwrite and self.has_split(cat_id):
            return
        template_ids = set(get_template_ids(cat_id))
        example_ids = get_example_ids(cat_id)
        example_ids = [i for i in example_ids if i not in template_ids]
        example_ids.sort()
        random.seed(self._seed)
        random.shuffle(example_ids)
        train_ids, eval_ids = _train_eval_partition(
            example_ids, self._train_prop)
        train_ids.sort()
        eval_ids.sort()
        for mode, ids in (('train', train_ids), ('eval', eval_ids)):
            with open(self.get_txt_path(cat_id, mode), 'w') as fp:
                fp.writelines(('%s\n' % i for i in ids))


def _train_eval_partition(example_list, train_prop=0.8):
    n = len(example_list)
    n_train = int(n*train_prop)
    return example_list[:n_train], example_list[n_train:]


def get_example_ids(cat_id, mode, **config_kwargs):
    return SplitConfig(**config_kwargs).get_example_ids(cat_id, mode)


if __name__ == '__main__':
    from template_ffd.templates.ids import get_templated_cat_ids
    config = SplitConfig()
    for cat_id in get_templated_cat_ids():
        config.create_split(cat_id)
