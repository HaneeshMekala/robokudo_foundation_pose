# Environment for running the FoundationPose RoboKudo pipeline on Tracy's machine.
#   usage:  source env_fp_tracy.sh
#
# Replaces the original 'env_fp.sh', which pointed at a '/home/student' user and
# a CUDA 11.8 system toolkit, neither of which exist here.
#
# Three differences from that script:
#   1. The workspace is '/home/tracy/ros2_ws', not '/home/student/ros_ws'.
#   2. There is no system CUDA toolkit.  The driver supports CUDA 13.0 but ships
#      no compiler, and 'nvidia-smi' showing "CUDA Version: 13.0" only reports
#      the driver's ceiling.  nvcc comes from pip wheels instead, assembled into
#      a CUDA_HOME tree by 'setup_cuda_home.sh'.
#   3. gcc 13 is the system compiler and CUDA 13 accepts it, so none of the
#      g++-11 pinning the old script did is needed.
#
# The venv is '~/.virtualenvs/robokudo', not a FoundationPose-specific one.  A
# dedicated venv was tried and abandoned: 'semantic_digital_twin' (via robokudo)
# pulls jax/scipy/ortools/rerun, which need numpy >= 2.0, while numba needs
# <= 2.2 - so an isolated env lands on numpy 2.2.6, and every apt-installed
# package compiled against numpy 1.x then fails with 'numpy.core.multiarray
# failed to import'.  The robokudo venv sits on numpy 1.26.4, consistent with
# the system, and already has the whole robokudo/semantic_digital_twin/giskardpy
# chain working.  Only the FoundationPose extras were added to it.
#
# CUDA_HOME still points into '.venv_fp', which is kept solely as the container
# for the pip CUDA toolkit ('setup_cuda_home.sh' builds the tree there).

export PATH=/usr/bin:$PATH

source /opt/ros/jazzy/setup.bash
source /home/tracy/ros2_ws/install/setup.bash

REPO=/home/tracy/robokudo_foundation_pose

# pip-provided CUDA toolkit, built by 'setup_cuda_home.sh'
export CUDA_HOME=$REPO/.venv_fp/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# the RTX 3080 is compute capability 8.6; naming it explicitly stops torch's
# JIT extensions from probing and building for architectures we do not have
export TORCH_CUDA_ARCH_LIST="8.6"

source /home/tracy/.virtualenvs/robokudo/bin/activate

echo "python : $(which python) ($(python -V 2>&1))"
echo "ros    : $ROS_DISTRO"
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "nvcc   : $($CUDA_HOME/bin/nvcc --version | tail -2 | head -1 | tr -s ' ')"
else
    echo "nvcc   : MISSING - run 'bash setup_cuda_home.sh' first"
fi
