#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${BRPLUS_ENV_NAME:-BeyondRetarget_plus}"
PYTHON_VERSION="${BRPLUS_PYTHON_VERSION:-3.10}"

command -v conda >/dev/null || { echo 'conda is required.' >&2; exit 1; }

# Conda activation hooks may legitimately read optional, unset variables.
# Keep nounset disabled only while executing Conda's own shell code.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" pip -y
fi

set +u
conda activate "$ENV_NAME"
set -u
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements_beyondretarget.txt"

if [[ "${BRPLUS_DOWNLOAD_SHARED_ASSETS:-1}" == "1" ]]; then
  python "$ROOT/scripts/download_base_shared_assets.py" \
    ${BRPLUS_SHARED_ASSET_FORCE:+--force}
else
  echo "Skipping shared Base robot/YOLO downloads (BRPLUS_DOWNLOAD_SHARED_ASSETS=0)."
fi

if [[ "${BRPLUS_FETCH_SUBMODULES:-1}" == "1" ]] && command -v git >/dev/null; then
  mkdir -p "$ROOT/thirdparty/DROID-SLAM/thirdparty"
  if [[ ! -d "$ROOT/thirdparty/DROID-SLAM/thirdparty/lietorch/lietorch" ]]; then
    git clone --depth 1 https://github.com/princeton-vl/lietorch.git \
      "$ROOT/thirdparty/DROID-SLAM/thirdparty/lietorch"
  fi
  if [[ ! -d "$ROOT/thirdparty/DROID-SLAM/thirdparty/eigen/Eigen" ]]; then
    git clone --depth 1 https://gitlab.com/libeigen/eigen.git \
      "$ROOT/thirdparty/DROID-SLAM/thirdparty/eigen"
  fi
fi

if [[ "${BRPLUS_INSTALL_FRONTEND:-0}" == "1" ]]; then
  python -m pip install \
    'git+https://github.com/facebookresearch/detectron2.git@a59f05630a8f205756064244bf5beb8661f96180' \
    'git+https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git'
fi

if [[ -n "${BRPLUS_ASSET_MANIFEST:-}" ]]; then
  python "$ROOT/scripts/download_plus_assets.py" --manifest "$BRPLUS_ASSET_MANIFEST" \
    ${BRPLUS_ASSET_FORCE:+--force}
elif [[ -n "${BRPLUS_ASSET_MANIFEST_URL:-}" ]]; then
  python "$ROOT/scripts/download_plus_assets.py" --manifest "$BRPLUS_ASSET_MANIFEST_URL" \
    ${BRPLUS_ASSET_FORCE:+--force}
else
  echo "No optional Plus asset manifest supplied; SMPL files and datasets remain external."
fi

# Verify both the installed distribution and the module actually imported by
# this interpreter after every dependency installation step. Never remove
# numpy/_core automatically: a mixed environment needs a clean reinstall,
# while NumPy 2.x pickles may need to be regenerated.
python - <<'PY'
import sys
from importlib import metadata

expected = "1.23.5"
try:
    distribution_version = metadata.version("numpy")
except metadata.PackageNotFoundError:
    distribution_version = "<not installed>"
try:
    import numpy
except Exception as exc:
    raise SystemExit(
        f"NumPy import failed in {sys.executable}: {exc!r}. "
        "Use a clean environment; do not delete numpy/_core manually."
    ) from exc
runtime_version = numpy.__version__
print(f"NumPy distribution: {distribution_version}")
print(f"NumPy runtime: {runtime_version} ({numpy.__file__})")
if distribution_version != expected or runtime_version != expected:
    raise SystemExit(
        f"NumPy mismatch: distribution={distribution_version}, runtime={runtime_version}, "
        f"expected={expected}. Use a clean environment; do not delete numpy/_core manually."
    )
PY

echo "Environment ready: $ENV_NAME"
echo "Next: conda activate $ENV_NAME"
echo "Then run: python scripts/check_installation.py"
