# RoboKudo FoundationPose

This repository contains the `FoundationPose` 6D pose estimator/tracker that can be used in RoboKudo.

The first package contains one RoboKudo `Annotator` that uses `FoundationPose` as a strong zero-shot object 6D pose estimator/tracker and the second one contains multiple 3D/CAD mesh models following the [CRAM conventions](https://ai.uni-bremen.de/wiki/3dmodeling/items) that are used in some pose/estimation examples.
The annotator can be used to estimate the 6D pose (position (in meter) and orientation) of prior detected objects in an RGB-D image, as long as for each object class a 3D/CAD model is available.
In detail, the annotator either estimates the 6D pose of an object represented by an `ObjectHypothesis` and updates the hypothesis with it, or if already a (coarse) 6D pose is available from the `ObjectHypothesis`, e.g. made by another estimator or the previous frame, it just tracked/refined, which much faster.
The estimator and tracker needs a prior object detector, to generate at least one `ObjectHypothesis`.
For estimation, it must contain a segmentation mask but not for tracking.

Based on: [FoundationPose](https://github.com/NVlabs/FoundationPose)


## Installation

<details><summary>Click to expand</summary>

Due to RoboKudos ROS 2 dependency Python 3.12 or higher required.

Open your virtual environment for RoboKudo

### 1. Package Installation

(Optional) Install missing `setuptools` with the following command:

```
pip3 install setuptools
```

Add a CUDA Toolkit 11.8, 12.6, 12.8 or higher (see [Link](https://developer.nvidia.com/cuda-12-8-1-download-archive)) and corresponding cuDNN 9 (see [Link](https://developer.nvidia.com/cudnn-downloads))

Note: You can install multiple different CUDA Toolkits following [this link](https://gist.github.com/garg-aayush/156ec6ddda3d62e2c0ddad00b7e66956).

Install PyTorch 2.2.0 or higher version that is compatible with the installed CUDA (see [Link](https://pytorch.org/get-started/locally/) for current or [Link](https://pytorch.org/get-started/previous-versions/) for older versions)

For example, you have CUDA 11.8 and want to install PyTorch 2.7.0, execute the command:

```
pip3 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

Install all remaining packages by executing the following commands:

```
pip3 install numpy trimesh open3d scipy opencv-python imageio omegaconf transformations
pip3 install kornia h5py warp-lang
pip3 install --no-build-isolation --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
pip3 install --no-build-isolation --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

Note: For CUDA <= 11.8 Version you may additionally need to call ```export NVCC_APPEND_FLAGS="-ccbin g++-11"```, otherwise the last two packages may fail to compile.


### 2. Example Data Preparation

Download the `2025-01-09-11-12-19-YCB-drill-cheezit-mustard-masterchef-domino-tomatosoup.bag` from [link](https://nc.uni-bremen.de/index.php/s/tBe47AJ6DNTDXJx) as all examples using its images. 

You can convert the old ROS 1 `.bag` files to ROS 2 `.db3` ones using the `rosbags` pip package.
To install it, execute the following command:

```
pip3 install rosbags
```

Then to convert a `.bag` file, use:

```
rosbags-convert --src "some_ros1_data.bag" --dst "some_ros2_data_bag"
```

For example, to convert our chosen `.bag` file open the virtual environment for RoboKudo in the download folder and use the command:

```
rosbags-convert --src "2025-01-09-11-12-19-YCB-drill-cheezit-mustard-masterchef-domino-tomatosoup.bag" --dst "ycbv_objects_1"
```

then play the converted file in a loop with the command:

```
ros2 bag play ycbv_objects_1/ --loop
```

</details>


## How to interact with the project

<details><summary>Click to expand</summary>


### Deploy the models in a RoboKudo AnalysisEngine

The `foundation_pose_annotator.py` file in the dir `robokudo_foundation_pose/robokudo_foundation_pose/annotator` implements the RoboKudo annotator, which allows you to easily deploy the model `FoundationPose` in RoboKudo.

To use the `FoundationPoseAnnotator` in your `AnalysisEngine`; do the following:


#### 1. Import the module in your AnalysisEngine

```python
from robokudo_foundation_pose.annotator.foundation_pose_annotator import FoundationPoseAnnotator
```


#### 2. Instantiate the `FoundationPoseAnnotator.Descriptor` class to configure the Annotator

```python
# create a 'Descriptor' object
fp_descriptor = FoundationPoseAnnotator.Descriptor()

# list of 3D/CAD mesh model files to use.
fp_descriptor.parameters.mesh_files = ["path/to/cad_1.obj", "path/to/cad_1.ply"]

# list of the corresponding object ids as used in the classification of the 'ObjectHypothesis'
fp_descriptor.parameters.mesh_obj_ids = [42, 35]
```

Take a look at the source code/files for further configuration parameters.


#### 3. Include the `FoundationPoseAnnotator` in the RoboKudo pipeline

To include the `FoundationPoseAnnotator` in your AnalysisEngine, create a `robokudo.pipeline.Pipeline` object and use its `add_children` function to add the Annotator:

```python
seq = robokudo.pipeline.Pipeline("yourPipelineName")

seq.add_children(
    [
        # ... other Annotators in your pipeline
        FoundationPoseAnnotator(descriptor=fp_descriptor),
        # ... more Annotators
    ]
)
```

</details>


## Examples

There are three example `AnalysisEngine` in the dir `robokudo_foundation_pose/robokudo_foundation_pose/descriptors/analysis_engines`.
The first two examples are in the `demo_single_object_cracker.py` and `demo_single_object_mustard.py` files, they apply the 6D pose estimation pipeline to all images with a single annotated `Cracker` and `Mustard` object, respectively.
The third example in the `demo_multiple_objects` file does the same but for multiple objects (`Cracker`, `Mustard` `Sugar` and `Tomato Soup`) simultaneously.

To start the examples, one can use the commands:

```
ros2 run robokudo main _ae=demo_single_object_cracker _ros_pkg=robokudo_foundation_pose
ros2 run robokudo main _ae=demo_single_object_mustard _ros_pkg=robokudo_foundation_pose
ros2 run robokudo main _ae=demo_multiple_objects _ros_pkg=robokudo_foundation_pose
```

Note: Make sure the `2025-01-09-11-12-19-YCB-drill-cheezit-mustard-masterchef-domino-tomatosoup.bag` file is currently played by ROS 2 (see also [Example Data Preparation](#2-example-data-preparation) for details).
