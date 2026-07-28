#!/usr/bin/env bash
# FastOPD baseline with a 1,024-token training response budget.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAX_RESPONSE_LENGTH=1024
exec bash "${SCRIPT_DIR}/../opd.sh" "$@"
