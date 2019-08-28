# fpointnet-tiny
Partial reimplementation of Frustum-PointNets, using the PointNet (not the ++ version) architecture.
This repository does not implement the box fitting part of Frustum-PointNets, only the part up to and including segmenting points into background and object.

# Installation
Install NumPy, SciPy, TensorFlow 2.0-beta1 using pip, currently rc0 does not work. 
For running with a GPU, install appropriate CUDA and CUDNN versions, this code was tested with CUDA 10.1 and CUDNN 7.6 installed via Conda.
