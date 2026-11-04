#!/usr/bin/env bash
# Build and install PyTorch3D (with its CUDA extension) into the robokudo venv.
#
#   usage:  bash install_pytorch3d.sh
#
# Why earlier attempts failed
# ---------------------------
# The README installs 'pytorch3d@stable'.  That tag (and the v0.7.9 release) is
# older than upstream commit f5f6b78 (2025-12-17, "Add initial CUDA 13.0 support
# for pulsar"). CUDA 13 changed the default ELF visibility of __global__
# functions and template stubs, so pulsar's explicitly instantiated
# 'pulsar::Renderer::render<true>' became invisible to the linker:
#
#     undefined reference to pulsar::Renderer::render<true>
#
# The fix passes '--device-entity-has-hidden-visibility=false
# -static-global-template-stub=false' to nvcc when torch is a CUDA >= 13 build.
# It exists only on 'main', so this script builds a pinned commit from main.
#
# Tested off-machine with the same stack: Python 3.12, gcc 13.3,
# torch 2.12.1+cu130, the pip CUDA toolkit 13.0.88 (via setup_cuda_home.sh),
# nvidia-cuda-cccl 13.0.85, TORCH_CUDA_ARCH_LIST=8.6.  The build takes ~9 min
# on 4 cores, links with 0 undefined pulsar symbols, and its so3_exp_map /
# rotation_6d_to_matrix agree with robokudo_foundation_pose/transforms.py.
#
# Rules this script keeps (see PROGRESS.md):
#   * the source is cloned OUTSIDE the repo - a 'pytorch3d/' folder in the repo
#     root once shadowed the real package as an empty namespace package;
#   * every CUDA wheel matches the 13.0 toolkit (cccl pinned to 13.0.85);
#   * nothing is allowed to move numpy in the robokudo venv (numpy==1.26.4).
#
# Re-run it after any torch upgrade: the extension is built against one torch.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ROS's setup.bash does not survive 'set -u', so source the env first
# shellcheck disable=SC1091
source "$REPO/env_fp_tracy.sh"
set -euo pipefail

PYTORCH3D_REF="${PYTORCH3D_REF:-e4ad6e5}"           # tested commit on main
PYTORCH3D_SRC="${PYTORCH3D_SRC:-$HOME/src/pytorch3d}"
DIST="${PYTORCH3D_DIST:-$HOME/src/pytorch3d-dist}"
FP_PIP="$REPO/.venv_fp/bin/pip"

case "$PYTORCH3D_SRC" in
    "$REPO"/*) echo "PYTORCH3D_SRC must be outside $REPO" >&2; exit 1 ;;
esac

echo "== checking the toolchain"
python - <<'EOF'
import sys, torch
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"gpu {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("torch sees no GPU - fix that before building")
EOF
nvcc --version | tail -2 | head -1

echo "== CUB / Thrust headers (nvidia-cuda-cccl) in the pip CUDA toolkit"
if [ ! -d "$CUDA_HOME/include/cccl/cub" ]; then
    "$FP_PIP" install --no-deps "nvidia-cuda-cccl==13.0.85"
    bash "$REPO/setup_cuda_home.sh"
fi
[ -d "$CUDA_HOME/include/cccl/cub" ] || { echo "no cub under $CUDA_HOME/include/cccl" >&2; exit 1; }
export CUB_HOME="$CUDA_HOME/include/cccl"

echo "== source: $PYTORCH3D_SRC @ $PYTORCH3D_REF"
if [ ! -d "$PYTORCH3D_SRC/.git" ]; then
    mkdir -p "$(dirname "$PYTORCH3D_SRC")"
    git clone https://github.com/facebookresearch/pytorch3d.git "$PYTORCH3D_SRC"
fi
git -C "$PYTORCH3D_SRC" fetch --quiet origin
git -C "$PYTORCH3D_SRC" checkout --quiet "$PYTORCH3D_REF"
grep -q "static-global-template-stub" "$PYTORCH3D_SRC/setup.py" \
    || { echo "$PYTORCH3D_REF predates the CUDA 13 pulsar fix" >&2; exit 1; }
rm -rf "$PYTORCH3D_SRC/build"     # stale objects from an older attempt

echo "== building the wheel (about 10 minutes)"
export FORCE_CUDA=1                      # build the CUDA kernels, fail if we cannot
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"  # lower this if the machine runs out of RAM
mkdir -p "$DIST"
rm -f "$DIST"/pytorch3d-*.whl
(cd "$PYTORCH3D_SRC" && python -m pip wheel --no-build-isolation --no-deps -w "$DIST" .)

echo "== installing into $(python -c 'import sys; print(sys.prefix)')"
python -m pip install --no-deps --force-reinstall "$DIST"/pytorch3d-*.whl
# iopath is pytorch3d's only runtime dependency; numpy is pinned so it cannot move
python -m pip install iopath "numpy==1.26.4"

echo "== verifying"
cd /    # import from site-packages, not from a source tree
python - <<'EOF'
import torch
import numpy, pytorch3d, pytorch3d._C
from pytorch3d.ops import knn_points
from pytorch3d.renderer.points.pulsar import Renderer
from pytorch3d.transforms import so3_exp_map, rotation_6d_to_matrix

print(f"pytorch3d {pytorch3d.__version__} at {pytorch3d.__file__}")
assert numpy.__version__ == "1.26.4", f"numpy moved to {numpy.__version__}"

# a real CUDA kernel from the extension, not just an import
p = torch.rand(1, 500, 3, device="cuda")
d = knn_points(p, p, K=1).dists
assert d.abs().max().item() < 1e-6, "knn_points on CUDA returned nonsense"
print("knn_points (CUDA kernel) OK")

Renderer(width=64, height=64, max_num_balls=10).cuda()
print("pulsar Renderer OK")

v = torch.randn(1000, 3, device="cuda")
R = so3_exp_map(v)
I = torch.eye(3, device="cuda").expand_as(R)
assert (R @ R.transpose(1, 2) - I).abs().max().item() < 1e-5
assert rotation_6d_to_matrix(torch.randn(10, 6, device="cuda")).shape == (10, 3, 3)
print("transforms OK")
print("pytorch3d is installed and working")
EOF
