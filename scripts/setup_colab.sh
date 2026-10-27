#!/usr/bin/env bash
# Prepare an isolated environment on a GPU-enabled Colab runtime.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work_root="${DARK_EVGS_WORKDIR:-/content}"
test -d "$work_root" || { echo 'Choose an existing DARK_EVGS_WORKDIR (Colab defaults to /content).' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'Select a GPU runtime in Colab first.' >&2; exit 1; }
command -v uv >/dev/null || python -m pip install uv

if [[ ! -x /usr/local/cuda-12.6/bin/nvcc ]]; then
  # Colab images may ship a CUDA-enabled torch wheel without the compiler.
  # Add compiler/runtime development components only; do not install drivers.
  source /etc/os-release
  [[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || {
    echo 'Automatic compiler installation supports Ubuntu 24.04 only; install a compatible CUDA 12 toolkit manually.' >&2
    exit 1
  }
  [[ "$(id -u)" == 0 ]] || { echo 'CUDA compiler package installation requires root on this Colab runtime.' >&2; exit 1; }
  keyring_dir="$(mktemp -d)"
  trap 'rm -rf -- "$keyring_dir"' EXIT
  curl --fail --location --retry 2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o "$keyring_dir/cuda-keyring.deb"
  dpkg -i "$keyring_dir/cuda-keyring.deb"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends cuda-nvcc-12-6 cuda-cudart-dev-12-6 cuda-cccl-12-6
fi
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export DARK_EVGS_PYTHON="${DARK_EVGS_PYTHON:-3.12}"
uv python install "$DARK_EVGS_PYTHON"
export DARK_EVGS_VENV="${DARK_EVGS_VENV:-$work_root/.venvs/dark-evgs}"
export DARK_EVGS_CACHE="${DARK_EVGS_CACHE:-$work_root/.cache/dark-evgs}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$work_root/.cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$work_root/.cache/pip}"
export MAX_JOBS="${MAX_JOBS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0))')}"
if [[ -n "${keyring_dir:-}" ]]; then
  rm -f -- "$keyring_dir/cuda-keyring.deb"
  rmdir -- "$keyring_dir"
  trap - EXIT
fi
exec bash "$repo_root/scripts/setup_reproduction.sh" cuda
