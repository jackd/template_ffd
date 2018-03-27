import os

_eval_dir = os.path.realpath(os.path.dirname(__file__))


def get_eval_dir(*args):
    folder = os.path.join(_eval_dir, '_eval', *args)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


def get_eval_path(*args):
    return os.path.join(get_eval_dir(*args[:-1]), args[-1])
