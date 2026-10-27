#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
cd "$(dirname "$0")/.."
TRAM_PYTHON="${TRAM_PYTHON:-/nasdata/tianyu/conda_env/tram_beyondretarget/bin/python}"
if [[ -s data/pascal_occluders.pkl ]]; then
  echo 'Already present: data/pascal_occluders.pkl (not overwritten)'
  exit 0
fi
"$TRAM_PYTHON" -c 'import gdown' || {
  echo "Install in the independent TRAM environment: $TRAM_PYTHON -m pip install gdown"
  exit 1
}
mkdir -p data
# Official TRAM scripts/download_pretrain.sh URL; download to a partial file.
"$TRAM_PYTHON" -m gdown --continue --timeout 30 --retries 1 -O data/pascal_occluders.pkl.partial \
  'https://drive.google.com/file/d/1_Qv9eAKVkfvZjdl9qaRyxVrIeAnAwevE/view?usp=sharing'
"$TRAM_PYTHON" - <<'PY'
from pathlib import Path
p=Path('data/pascal_occluders.pkl.partial')
with p.open('rb') as stream:
    head=stream.read(256).lower()
if p.stat().st_size<100000 or b'<!doctype html' in head or b'<html' in head:
    raise RuntimeError('Download is incomplete or an HTML error page; kept .partial for recovery')
p.rename('data/pascal_occluders.pkl')
print('Saved data/pascal_occluders.pkl (official synthetic occlusion augmentation resource)')
PY
