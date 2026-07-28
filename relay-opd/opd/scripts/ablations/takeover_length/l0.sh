#!/usr/bin/env bash
# Takeover-length ablation: replace only the trigger token.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RELAY_OPD_PARAGRAPHS_PER_TAKEOVER=0
exec bash "${SCRIPT_DIR}/../../relay_opd/train.sh" "$@"
