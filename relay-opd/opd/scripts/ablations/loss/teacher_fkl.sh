#!/usr/bin/env bash
# Loss ablation: top-128 teacher-renormalized FKL on takeover tokens.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOSS_MODE=relay_opd_fkl
export RELAY_ACTION_TOKEN=emitted
export VERL_DISTILLATION_DUAL_LOGPROBS=1
exec bash "${SCRIPT_DIR}/../../relay_opd/train.sh" \
    distillation.distillation_loss.topk=128 \
    +distillation.distillation_loss.relay_teacher_fkl_coef=1.0 \
    "$@"
