#!/usr/bin/env bash
# Relay trigger threshold ablation: teacher token must be outside student top-10.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RELAY_OPD_TRIGGER_TOPK=10
exec bash "${SCRIPT_DIR}/../../relay_opd/train.sh" "$@"
