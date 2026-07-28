#!/usr/bin/env bash
# TRD rewrite-trajectory FKL baseline.
#
# Student trains on:
#   original problem + empty-think student prompt -> teacher rewrite y_r
# Teacher top-k/FKL is scored under:
#   TRD rewrite prompt(problem + student y_o + empty think) -> same y_r
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_DATA=${TRAIN_DATA:?TRAIN_DATA is required}
export OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
export EXP_ID=${EXP_ID:-$(basename "${OUTPUT_DIR}")}
export LOSS_MODE=${LOSS_MODE:-forward_kl_topk_teacher_renorm}
export DISTILL_TOPK=${DISTILL_TOPK:-128}
export OFFLINE_DATA_SOURCE=${OFFLINE_DATA_SOURCE:-trd_rewrite_student_nothink_prompt}
export TEACHER_PROMPT_TOKEN_KEY=${TEACHER_PROMPT_TOKEN_KEY:-teacher_prompt_token_ids}

# TRD teacher scoring prompt includes the original student trajectory, so it can
# be much longer than the student training prompt. Keep actor/student context at
# 2k+16k while allowing the teacher scorer to prefill rewrite_prompt+y_r.
export TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-40960}
export TEACHER_MAX_NUM_BATCHED_TOKENS=${TEACHER_MAX_NUM_BATCHED_TOKENS:-4096}

exec bash "${SCRIPT_DIR}/seqkd.sh" "$@"
