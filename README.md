# fpointnet-tiny
Partial reimplementation of Frustum-PointNets, using the PointNet (not the ++ version) architecture.
This repository does not implement the box fitting part of Frustum-PointNets, only the part up to and including segmenting points into background and object.

# Installation
Install NumPy, SciPy, TensorFlow 2.0-beta1 using pip, currently rc0 does not work. 
For running with a GPU, install appropriate CUDA and CUDNN versions, this code was tested with CUDA 10.1 and CUDNN 7.6 installed via Conda.

# Usage
To train the model the initial data should already be separated into different frustums. Currently each frustum should have its own file
as a `.npz` file containing `points` and `class_name`. If the initial frustums data is not normalized, it should be normalized via `preprocess.py`.
The training script requires both training and validation data to be passed as separate directories containing the aforementioned files.

In addition to training there is a prediction/evaluation program, which takes as input unnormalized frustums data and performs normalization on its own. This file requires points to be labelled, but can be easily altered to not require it for new predictions.