#!/usr/bin/env bash
# Relay budget ablation: stop after three teacher takeovers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RELAY_OPD_MAX_TAKEOVERS=3
exec bash "${SCRIPT_DIR}/../../relay_opd/train.sh" "$@"
