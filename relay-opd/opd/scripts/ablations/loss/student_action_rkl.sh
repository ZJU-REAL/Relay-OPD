#!/usr/bin/env bash
# Loss ablation: score the student's discarded draft action on takeover tokens.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOSS_MODE=relay_opd
export RELAY_ACTION_TOKEN=student_draft
exec bash "${SCRIPT_DIR}/../../relay_opd/train.sh" "$@"
