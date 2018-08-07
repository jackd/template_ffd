# template_ffd
Code for paper [Learning Free-Form Deformations for 3D Object Reconstruction](https://arxiv.org/abs/1803.10932) in [this repository](https://github.com/jackd/template_ffd).

# Getting Started
```
cd /path/to/parent_dir
git clone https://github.com/jackd/template_ffd.git
# non-pip dependencies
git clone https://github.com/jackd/dids.git                  # framework for manipulating datasets
git clone https://github.com/jackd/util3d.git                # general 3d object utilities
git clone https://github.com/jackd/shapenet.git              # dataset access
git clone https://github.com/jackd/tf_nearest_neighbour.git  # for chamfer loss
git clone https://github.com/jackd/tf_toolbox.git            # optional
```
To run, ensure the parent directoy is on your `PYTHON_PATH`.
```
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```

So long as your `PYTHONPATH` is set as above, these repositories should work 'out of the box', except for `tf_nearest_neighbour` which requires the tensorflow op to be built. See the main [repository](https://github.com/jackd/tf_nearest_neighbour) for details.

Install pip dependencies
```
pip install h5py progress numpy pyemd
```

To use visualizations you'll also need `mayavi`.
```
pip install mayavi
```

See [tensorflow documentation](https://www.tensorflow.org/install/) for installation. CUDA enabled GPU recommended.

## Data
This repository depends on the Dictionary Interface to Datasets ([`dids`](https://github.com/jackd/dids.git)) repository for dataset management and [`util3d`](https://github.com/jackd/util3d.git) for various 3d utility functions.

This code base is set up to train on the [`ShapeNet`](https://www.shapenet.org/) Core dataset. We cannot provide the data for this dataset, though it is freely available for registered users. We provide functionality for rendering, loading and converting data in the [`shapenet`](https://github.com/jackd/shapenet) repository. For this project, most data accessing should "just work". There are, however, 2 manual steps that must be completed.

1. Add the path to your shapenet core data to the environment variable `SHAPENET_CORE_PATH`,
```
export SHAPENET_CORE_PATH=/path/to/shapenet/dataset/ShapeNetCore.v1
```
This folder should contain the `.zip` files for each category, named by the category id, e.g. all plane `obj` files should be in `02691156.zip`.
2. Render the images,
```
cd /path/to/parent_dir/shapenet/core/blender_renderings/scripts
python render_cat.py plane
python create_archive.py plane
```
[`Blender`](https://www.blender.org/) is required for this. The binary must either be on your path, or supplied via `render_cat.py`'s `--blender_path` argument.

Other data preprocessing is required before training can begin (parsing mesh data, sampling meshes, calculating FFD decomposition), though this should be handled as the need arises.

In order to evaluate IoU scores, meshes must first be converted to voxels. To allow this, make the `util3d` binvox binary executable
```
chmod +x /path/to/parent_dir/util3d/bin/binvox
```

You can force any of this data processing for any category to occur by manually. See the example for generating plane data below.
```
cd /path/to/parent_dir/shapenet/core/meshes/scripts
python generate_mesh_data.py plane
cd ../../point_clouds/scripts
python create_point_clouds.py plane
cd ../../voxels/scripts
# For IoU data.
python create_voxels.py plane
python create_archive.py plane
```

Note evaluation of models produces a large amount of data. In particular, inferred meshes generated for IoU evaluation can be particularly large for a low `edge_length_threshold`. You can safely delete any data in `inference/_inferences` or `eval/_eval` and it will be regenerated if required.

## Models
Different models can be built using different hyper-parameter sets. Models are built using the `model.template_ffd_builder.TemplateFfdBuilder` class. Each hyperparameter set should have a `MODEL_ID` and an associated `model/params/MODEL_ID.json` file. Default values are speficied where they are used in the code.

See `paper/create_paper_params.py` for the parameter sets used for the models presented in the paper.

## Training
Training can be done via the `scripts/train.py` script. For example,
```
python train.py example -s 200000
```
will train the model with ID `'example'` for 200000 steps (default is 100000).

To view training summaries, run
```
tensorboard --logdir=model/_model/MODEL_ID
```

Training to 100000 steps as done in the paper takes roughly 8 hours to an NVidia GTX-1070.

## Evaluation
There are a number of steps to evaluation, depending on the metrics required.
* To create predictions (network outputs, deformation parameters `Delta P`), run `scripts/infer.py MODEL_ID`
* See also `scripts/iou.py`, `scripts/chamfer.py` and `scripts/ffd_emd.py` (slow).

## Paper Figures
See the `paper` subdirectory for various scripts used to generate the figures presented in the paper.

## Reference
If you find this code useful in your research, please cite the [following paper](https://128.84.21.199/abs/1803.10932).
```
@article{jack2018learning,
  title={Learning Free-Form Deformations for 3D Object Reconstruction},
  author={Jack, Dominic and Pontes, Jhony K and Sridharan, Sridha and Fookes, Clinton and Shirazi, Sareh and Maire, Frederic and Eriksson, Anders},
  journal={arXiv preprint arXiv:1803.10932},
  year={2018}
}
```

## CHANGELOG
Since the initial release, a small bug has been fixed where batch normalization was being applied both before and after activations in some cases. This shouldn't make a massive difference to performance, but may mean models previously trained can no longer be loaded properly. To revert to older functionality, add `'use_bn_bugged_version': true` to the params file.
