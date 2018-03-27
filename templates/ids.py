def _get_template_ids():
    import json
    import path
    with open(path.template_ids_path, 'r') as f:
        template_ids = json.load(f)
    return template_ids


_template_ids = {k: tuple(v) for k, v in _get_template_ids().items()}


def get_template_ids(cat_id):
    return _template_ids[cat_id]


def get_templated_cat_ids():
    return _template_ids.keys()
