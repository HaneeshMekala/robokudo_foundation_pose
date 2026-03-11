Python 3.12 or higher (because of ROS 2)

## Installation

<details><summary>Click to expand</summary>


### 1. Package Installation

Open your virtual environment for RoboKudo

(Optional) Add missing "setuptools"

```
pip3 install setuptools
```

(Optional) Add a CUDA Toolkit 11.8, 12.6, 12.8 or higher (see [Link](https://developer.nvidia.com/cuda-12-8-1-download-archive)) and corresponding cuDNN 9 (see [Link](https://developer.nvidia.com/cudnn-downloads))

(Optional) Install PyTorch 2.2.0 or higher version that is compatible with the insatlled CUDA (see [Link](https://pytorch.org/get-started/locally/) for current or [Link](https://pytorch.org/get-started/previous-versions/) or older versions)

Examply to you have CUDA 11.8 and want to install PyTorch 2.7.0, executing the following command:

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

</details>


