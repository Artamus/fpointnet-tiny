# fpointnet-tiny
Partial reimplementation of Frustum-PointNets, using the PointNet (not the ++ version) architecture.
This repository does not implement the box fitting part of Frustum-PointNets, only the part up to and including segmenting points into background and object.

# Installation
Install NumPy, SciPy, TensorFlow 2.0-beta1 using pip, currently rc0 does not work. 
For running with a GPU, install appropriate CUDA and CUDNN versions, this code was tested with CUDA 10.1 and CUDNN 7.6 installed via Conda.

# Usage
To train the model the initial data should already be separated into different frustums. Currently each frustum should have its own file
as a `.npz` file containing `points` and `class_name`.
The training script requires both training and validation data to be passed as directories containing the aforementioned files.

TODO: Prediction/evaluation