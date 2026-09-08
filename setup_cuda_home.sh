#!/usr/bin/env bash
# Assemble a CUDA_HOME tree from the pip-installed CUDA toolkit.
#
# There is no system CUDA toolkit on this machine and no root is needed:
# 'cuda-toolkit[nvcc]' ships a complete toolkit (bin/, include/, lib/, nvvm/)
# inside site-packages.  Two things have to be fixed up for it to work as a
# CUDA_HOME:
#
#   1. torch's cpp_extension and nvdiffrast's JIT build expect 'lib64', while
#      the wheel calls it 'lib'.
#   2. The wheels ship only versioned sonames ('libcudart.so.13'), so a plain
#      '-lcudart' fails to link.  Unversioned aliases are added here.
#
# Component versions must agree or nvcc emits PTX that its own ptxas rejects
# ("Unsupported .version 9.3; current version is '9.0'"); nvvm and cuda-crt are
# pinned to the nvcc version for that reason.
#
# Safe to re-run; it replaces the tree it made.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="$REPO/.venv_fp/lib/python3.12/site-packages/nvidia"
SRC="$SITE/cu13"
DST="$REPO/.venv_fp/cuda"

[ -x "$SRC/bin/nvcc" ] || { echo "no nvcc at $SRC - is 'cuda-toolkit[nvcc]' installed?" >&2; exit 1; }

rm -rf "$DST"
mkdir -p "$DST/lib64"
for d in bin include nvvm; do
    ln -s "$SRC/$d" "$DST/$d"
done

# link every shared/static library the toolkit wheels provide, then add an
# unversioned alias for each versioned soname so '-l<name>' resolves
for lib in "$SRC"/lib/*; do
    ln -sf "$lib" "$DST/lib64/$(basename "$lib")"
done
for so in "$DST"/lib64/*.so.*; do
    [ -e "$so" ] || continue
    base="$(basename "$so")"
    stem="${base%%.so.*}"
    [ -e "$DST/lib64/$stem.so" ] || ln -s "$so" "$DST/lib64/$stem.so"
done
ln -s lib64 "$DST/lib"

echo "CUDA_HOME ready at $DST"
"$DST/bin/nvcc" --version | tail -2 | head -1
