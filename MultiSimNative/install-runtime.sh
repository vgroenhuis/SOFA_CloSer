#!/bin/bash
# Installs the SOFA runtime into ./runtime (no root needed):
#   - SOFA v26.06 binary release (includes SoftRobots)
#   - a Python 3.12 venv (via uv) with the orchestrator + sofaweb dependencies
# Re-running is safe; finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")"
SOFA_VERSION="${SOFA_VERSION:-26.06.00}"
RT="$PWD/runtime"
mkdir -p "$RT/dl"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/ (curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; }

if [ ! -d "$RT/sofa/plugins/SofaPython3" ]; then
	zip="$RT/dl/SOFA_v${SOFA_VERSION}_Linux_Python3.12.zip"
	[ -s "$zip" ] || curl -fL -o "$zip" "https://github.com/sofa-framework/sofa/releases/download/v${SOFA_VERSION}/SOFA_v${SOFA_VERSION}_Linux_Python3.12.zip"
	echo "=== Extracting SOFA"
	rm -rf "$RT/x"
	# Not `unzip`: the archive contains symlinks (and unzip may be missing).
	python3 -I - "$zip" "$RT/x" <<'PY'
import os, stat, sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); dest = sys.argv[2]
for i in z.infolist():
    mode = i.external_attr >> 16
    p = os.path.join(dest, i.filename)
    if i.is_dir():
        os.makedirs(p, exist_ok=True); continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if stat.S_ISLNK(mode):
        if os.path.lexists(p): os.remove(p)
        os.symlink(z.read(i).decode(), p)
    else:
        with open(p, "wb") as f: f.write(z.read(i))
        if mode: os.chmod(p, stat.S_IMODE(mode))
PY
	mv "$RT"/x/SOFA_* "$RT/sofa" && rmdir "$RT/x"
fi

if [ ! -x "$RT/venv/bin/python" ]; then
	echo "=== Creating Python 3.12 venv"
	uv venv --python 3.12 "$RT/venv"
fi
uv pip install --python "$RT/venv/bin/python" -r orchestrator/requirements.txt "numpy>=1.24,<2"

echo "=== Checking SOFA imports"
PYLIB=$("$RT/venv/bin/python" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHONPATH="$RT/sofa/plugins/SofaPython3/lib/python3/site-packages" \
LD_LIBRARY_PATH="$RT/sofa/lib:$RT/sofa/plugins/SofaPython3/lib:$PYLIB" \
	"$RT/venv/bin/python" -c "import Sofa, numpy; print('SOFA python bindings OK')" 2>&1 | tail -1
echo "Runtime ready in $RT"
