# Environment for running the FoundationPose RoboKudo pipeline on this machine.
#   usage:  source env_fp.sh
#
# Three things this fixes:
#   1. /usr/local/bin/python3 is a symlink to Python 3.8 and shadows the 3.12
#      that RoboKudo is installed for - put /usr/bin first.
#   2. The CUDA toolkit here is 11.8, which refuses gcc > 11, so point nvcc at
#      g++-11 (needed whenever a torch extension is rebuilt).
#   3. ROS 2 Jazzy and the colcon workspace have to be sourced before the venv
#      so robokudo/rclpy/py_trees resolve.
#
# Activating the venv last is what makes `ros2 run robokudo main` work: the
# launcher's shebang is `#!/usr/bin/env python3`, so it takes whichever python3
# is first on PATH - which is now the venv's, not the 3.8 shim.

export PATH=/usr/bin:$PATH

source /opt/ros/jazzy/setup.bash
source /home/student/ros_ws/install/setup.bash

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export NVCC_APPEND_FLAGS="-ccbin g++-11"
export CC=gcc-11
export CXX=g++-11

source /home/student/robokudo_foundation_pose/.venv_fp/bin/activate

echo "python : $(which python) ($(python -V 2>&1))"
echo "ros    : $ROS_DISTRO"
echo "cuda   : $($CUDA_HOME/bin/nvcc --version | tail -2 | head -1 | tr -s ' ')"
