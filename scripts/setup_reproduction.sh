#!/usr/bin/env bash
# Independent reproduction environment; original environment.yml is left intact.
set -euo pipefail
mode="${1:-cpu}"
case "$mode" in
  cpu) torch_variant=cpu ;;
  cuda) torch_variant="${DARK_EVGS_CUDA_VARIANT:-cu121}" ;;
  *) echo "Usage: bash scripts/setup_reproduction.sh [cpu|cuda]" >&2; exit 2 ;;
esac
case "$torch_variant" in cpu|cu118|cu121) ;; *) echo 'Choose cu118 or cu121 for DARK_EVGS_CUDA_VARIANT.' >&2; exit 2 ;; esac
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${DARK_EVGS_VENV:-/workspace/.venvs/dark-evgs}"
cache_dir="${DARK_EVGS_CACHE:-/workspace/.cache/dark-evgs}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
mkdir -p "$cache_dir" "$UV_CACHE_DIR" "$PIP_CACHE_DIR"
command -v uv >/dev/null || { echo 'Install uv from its official distribution, or create this environment manually as documented.' >&2; exit 1; }
uv venv --allow-existing "$venv_dir" --python "${DARK_EVGS_PYTHON:-python3}"
uv pip install --python "$venv_dir/bin/python" --index-url "https://download.pytorch.org/whl/$torch_variant" torch==2.5.1
if [[ "${DARK_EVGS_LPIPS:-0}" == 1 ]]; then
  uv pip install --python "$venv_dir/bin/python" --index-url "https://download.pytorch.org/whl/$torch_variant" torchvision==0.20.1
  uv pip install --python "$venv_dir/bin/python" lpips==0.1.4
fi
uv pip install --python "$venv_dir/bin/python" -r "$repo_root/requirements-reproduction.txt" 'setuptools==75.6.0' 'wheel==0.45.1' 'ninja==1.11.1.3'
if [[ "$mode" == cuda ]]; then
  command -v nvcc >/dev/null || { echo 'CUDA training needs a local CUDA toolkit and nvcc, in addition to PyTorch wheels.' >&2; exit 1; }
  "$venv_dir/bin/python" -c 'import torch; assert torch.cuda.is_available(), "No usable NVIDIA GPU/driver"; print("GPU:", torch.cuda.get_device_name(0), "PyTorch CUDA:", torch.version.cuda)'
  nvcc --version
  ext_dir="$cache_dir/diff-gaussian-rasterization-59f5f77"
  ext_commit=59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
  if [[ ! -d "$ext_dir" ]]; then
    git clone --no-checkout https://github.com/graphdeco-inria/diff-gaussian-rasterization.git "$ext_dir"
    git -C "$ext_dir" checkout --detach "$ext_commit"
  fi
  [[ "$(git -C "$ext_dir" rev-parse HEAD)" == "$ext_commit" ]] || { echo "Unexpected extension revision at $ext_dir; choose a fresh DARK_EVGS_CACHE." >&2; exit 1; }
  [[ -z "$(git -C "$ext_dir" status --porcelain --untracked-files=no)" ]] || { echo 'CUDA extension source has local changes; preserve them and choose a fresh cache.' >&2; exit 1; }
  git -C "$ext_dir" submodule update --init --recursive
  export MAX_JOBS="${MAX_JOBS:-4}"
  uv pip install --python "$venv_dir/bin/python" --no-build-isolation "$ext_dir"
  "$venv_dir/bin/python" -c 'from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings; print("CUDA rasterizer import passed")'
fi
uv pip check --python "$venv_dir/bin/python"
cd "$repo_root"
"$venv_dir/bin/python" -m pytest -q tests
printf 'Activate with: source %s/bin/activate\n' "$venv_dir"
