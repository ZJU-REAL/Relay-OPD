#!/usr/bin/env bash
# FastOPD baseline with an 8,192-token training response budget.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAX_RESPONSE_LENGTH=8192
exec bash "${SCRIPT_DIR}/../opd.sh" "$@"
