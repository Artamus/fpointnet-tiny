# fpointnet-tiny
Partial reimplementation of Frustum-PointNets, using the PointNet (not the ++ version) architecture.
This repository does not implement the box fitting part of Frustum-PointNets, only the part up to and including segmenting points into background and object.

# Installation
Install NumPy, SciPy, TensorFlow 2.0-beta1 using pip, currently rc0 does not work. 
For running with a GPU, install appropriate CUDA and CUDNN versions, this code was tested with CUDA 10.1 and CUDNN 7.6 installed via Conda.

# Data format
This existing code is written with the assumption that different scenes are numbered (like they are in KITTI) and that the data has already been divided into training, validation (and test) sets. The points for each frustum should be contained in a separate file with a file format of `{scene_id}_{frustum_id}.npz`. Each of these files should contain `points` (a (None, 4) NumPy ndarray) and `class_name` describing the class, allowed values are `person` and `car`, but this is easily extendable. The point of this is to be able to train on a single class when the frustums data contains data from multiple ones.

Training and validation (and test) data should live in separate folders.

# Usage
If the initial frustums data is not normalized, it should be normalized via `preprocess.py`.

The model can be trained via the training program. Output models are saved into the appropriately named models directory, with each run having its own subfolder. The models are only saved if validation accuracy improves and contain the epoch number of their creation. Therefore the model with the highest epoch number is the best.

In addition to training there is a prediction/evaluation program, which takes as input unnormalized frustums data and performs normalization on its own. This program requires points to be labelled, but can be easily altered to not require it to make predictions on not-yet-labelled data.