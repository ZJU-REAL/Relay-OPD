#!/usr/bin/env bash
# Loss-control run: k1 RKL on emitted tokens, including teacher takeovers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOSS_MODE=relay_opd
export RELAY_ACTION_TOKEN=emitted
exec bash "${SCRIPT_DIR}/../../relay_opd/train.sh" "$@"
