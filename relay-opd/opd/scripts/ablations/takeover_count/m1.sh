#!/usr/bin/env bash
# Relay budget ablation: stop after one teacher takeover.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RELAY_OPD_MAX_TAKEOVERS=1
exec bash "${SCRIPT_DIR}/../../relay_opd/train.sh" "$@"
