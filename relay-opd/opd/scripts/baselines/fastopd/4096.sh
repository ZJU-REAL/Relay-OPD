#!/usr/bin/env bash
# FastOPD baseline with a 4,096-token training response budget.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAX_RESPONSE_LENGTH=4096
exec bash "${SCRIPT_DIR}/../opd.sh" "$@"
