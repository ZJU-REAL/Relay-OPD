#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3.12}
VENV_DIR=${VENV_DIR:-${REPO_ROOT}/.venv}

if [[ -e "${VENV_DIR}" ]]; then
    echo "Refusing to modify an existing environment: ${VENV_DIR}" >&2
    echo "Set VENV_DIR to a new path and rerun." >&2
    exit 10
fi

python_version="$("${PYTHON_BIN}" -c 'import platform; print(platform.python_version())')"
if [[ "${python_version}" != 3.12.* ]]; then
    echo "Relay-OPD requires Python 3.12; found ${python_version}." >&2
    exit 11
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install \
    "pip==26.1.2" \
    "setuptools==80.10.2" \
    "wheel==0.47.0"

# The lock contains every transitive package from the clean validated
# environment. --no-deps is intentional: verl pins NumPy below 2 while
# vLLM's OpenCV dependency metadata requests NumPy 2, although the validated
# NumPy 1.26/OpenCV 4.13 pair is runtime-compatible.
python -m pip install --no-deps -r "${SCRIPT_DIR}/requirements.lock.txt"
python -m pip install --no-deps -e "${REPO_ROOT}"

RELAY_OPD_STRICT_VERSIONS=1 python "${SCRIPT_DIR}/verify_install.py"

echo
echo "Relay-OPD locked reference environment created at ${VENV_DIR}"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
