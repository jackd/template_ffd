import os

inference_dir = os.path.realpath(os.path.dirname(__file__))


def get_inference_subdir(*args):
    subdir = os.path.join(inference_dir, '_inferences', *args)
    if not os.path.isdir(subdir):
        os.makedirs(subdir)
    return subdir


def get_inference_path(*args):
    return os.path.join(get_inference_subdir(*args[:-1]), args[-1])
