#!/usr/bin/python


def clear_results(model_id, actual=False):
    import os
    import shutil
    from template_ffd.inference.predictions import get_predictions_data_path
    from template_ffd.inference.meshes import InferredMeshManager
    from template_ffd.inference.clouds import get_cloud_manager
    from template_ffd.inference.voxels import get_voxel_subdir
    from template_ffd.eval.chamfer import \
        get_chamfer_manager, get_template_chamfer_manager
    from template_ffd.eval.iou import IouAutoSavingManager
    from template_ffd.eval.ffd_emd import get_template_emd_manager
    from template_ffd.eval.ffd_emd import get_emd_manager

    def maybe_remove(path):
        if os.path.isfile(path):
            print('Removing file %s' % path)
            if actual:
                os.remove(path)
        elif os.path.isdir(path):
            print('Removing subdir %s' % path)
            if actual:
                shutil.rmtree(path)

    predictions_path = get_predictions_data_path(model_id)
    maybe_remove(predictions_path)
    maybe_remove(get_cloud_manager(model_id, pre_sampled=True).path)
    maybe_remove(get_chamfer_manager(model_id, pre_sampled=True).path)
    maybe_remove(get_template_chamfer_manager(model_id).path)
    maybe_remove(get_emd_manager(model_id, pre_sampled=True).path)
    maybe_remove(get_template_emd_manager(model_id).path)

    for elt in (None, 0.1, 0.05, 0.02, 0.01):
        maybe_remove(InferredMeshManager(model_id, elt).path)

        maybe_remove(get_cloud_manager(
            model_id, pre_sampled=False, edge_length_threshold=elt).path)

        for filled in (True, False):
            subdir = get_voxel_subdir(model_id, elt, filled=filled)
            maybe_remove(subdir)
            maybe_remove(IouAutoSavingManager(model_id, elt, filled).path)

        kwargs = dict(
            model_id=model_id, edge_length_threshold=elt)
        for ps in (True, False):
            maybe_remove(get_chamfer_manager(pre_sampled=ps, **kwargs).path)
            maybe_remove(get_emd_manager(pre_sampled=ps, **kwargs).path)

    if not actual:
        print('NOTE: this was a dry run. Files not actually removed')
        print('Use -a for actual run.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_id')
    parser.add_argument(
        '-a', '--actual_run', action='store_true', help='actual run')

    args = parser.parse_args()
    clear_results(args.model_id, args.actual_run)
