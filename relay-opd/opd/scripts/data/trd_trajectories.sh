#!/usr/bin/env bash
# Build TRD data: student trajectory, teacher rewrite, then tokenized parquet.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_OPD_DIR=${VERL_OPD_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
STUDENT_MODEL=${STUDENT_MODEL:?STUDENT_MODEL is required}
TEACHER_MODEL=${TEACHER_MODEL:?TEACHER_MODEL is required}
SOURCE_DATA=${SOURCE_DATA:?SOURCE_DATA is required}
OUTPUT_PARQUET=${OUTPUT_PARQUET:?OUTPUT_PARQUET is required}

OUTPUT_DIR=$(dirname "${OUTPUT_PARQUET}")
STUDENT_JSONL=${STUDENT_JSONL:-${OUTPUT_DIR}/trd_student_trajectories.jsonl}
REWRITE_JSONL=${REWRITE_JSONL:-${OUTPUT_PARQUET%.parquet}.jsonl}
WORK_ROOT=${WORK_ROOT:-${OUTPUT_DIR}/trd_generation.work}
STUDENT_GPU_GROUPS=${STUDENT_GPU_GROUPS:-0;1;2;3;4;5;6;7}
REWRITE_GPU_GROUPS=${REWRITE_GPU_GROUPS:-${STUDENT_GPU_GROUPS}}
MAX_NEW=${MAX_NEW:-16384}
STUDENT_MAX_MODEL_LEN=${STUDENT_MAX_MODEL_LEN:-$((2048 + MAX_NEW + 1))}
REWRITE_MAX_MODEL_LEN=${REWRITE_MAX_MODEL_LEN:-40960}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
SEED=${SEED:-0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
STUDENT_MAX_NUM_SEQS=${STUDENT_MAX_NUM_SEQS:-64}
STUDENT_MAX_NUM_BATCHED_TOKENS=${STUDENT_MAX_NUM_BATCHED_TOKENS:-65536}
REWRITE_MAX_NUM_SEQS=${REWRITE_MAX_NUM_SEQS:-16}
REWRITE_MAX_NUM_BATCHED_TOKENS=${REWRITE_MAX_NUM_BATCHED_TOKENS:-65536}
PYTHON_BIN=${PYTHON_BIN:-python3}

mkdir -p "${OUTPUT_DIR}" "${WORK_ROOT}/student_shards" "${WORK_ROOT}/rewrite_shards" "${WORK_ROOT}/logs"

active_pids=()
cleanup() {
    local pid
    for pid in "${active_pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup INT TERM

run_phase() {
    local mode=$1
    local model=$2
    local data=$3
    local shard_dir=$4
    local gpu_groups=$5
    local max_model_len=$6
    local max_num_seqs=$7
    local max_num_batched_tokens=$8
    local -a groups devices
    local shard group tp shard_out pid failed

    IFS=';' read -r -a groups <<< "${gpu_groups}"
    if ((${#groups[@]} == 0)); then
        echo "${mode}: GPU groups must not be empty" >&2
        exit 2
    fi

    active_pids=()
    for ((shard = 0; shard < ${#groups[@]}; shard++)); do
        group=${groups[shard]//[[:space:]]/}
        IFS=',' read -r -a devices <<< "${group}"
        tp=${#devices[@]}
        shard_out="${shard_dir}/shard_${shard}.jsonl"
        if [[ -f "${shard_out}" && -s "${shard_out}.summary.json" ]]; then
            echo "[trd-data] skip complete ${mode} shard ${shard}"
            continue
        fi

        (
            export CUDA_VISIBLE_DEVICES="${group}"
            export VLLM_ALLREDUCE_USE_FLASHINFER=0
            export VLLM_USE_FLASHINFER_SAMPLER=0
            export VLLM_ENABLE_V1_MULTIPROCESSING=0
            "${PYTHON_BIN}" "${VERL_OPD_DIR}/opd/data/generate_trd_trajectories.py" \
                --mode "${mode}" \
                --model "${model}" \
                --student-tokenizer "${STUDENT_MODEL}" \
                --data "${data}" \
                --out "${shard_out}" \
                --shard-id "${shard}" \
                --num-shards "${#groups[@]}" \
                --max-new "${MAX_NEW}" \
                --max-model-len "${max_model_len}" \
                --temperature "${TEMPERATURE}" \
                --top-p "${TOP_P}" \
                --seed "${SEED}" \
                --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
                --tensor-parallel-size "${tp}" \
                --max-num-seqs "${max_num_seqs}" \
                --max-num-batched-tokens "${max_num_batched_tokens}"
        ) >"${WORK_ROOT}/logs/${mode}_shard_${shard}.log" 2>&1 &
        active_pids+=("$!")
    done

    failed=0
    for pid in "${active_pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    active_pids=()
    if ((failed)); then
        echo "TRD ${mode} generation failed; inspect ${WORK_ROOT}/logs" >&2
        exit 1
    fi

    PHASE_SHARDS=${#groups[@]}
}

if [[ ! -s "${STUDENT_JSONL}" || ! -s "${STUDENT_JSONL}.summary.json" ]]; then
    echo "[trd-data] phase 1/3: sample student trajectories"
    run_phase \
        student "${STUDENT_MODEL}" "${SOURCE_DATA}" "${WORK_ROOT}/student_shards" \
        "${STUDENT_GPU_GROUPS}" "${STUDENT_MAX_MODEL_LEN}" \
        "${STUDENT_MAX_NUM_SEQS}" "${STUDENT_MAX_NUM_BATCHED_TOKENS}"
    "${PYTHON_BIN}" "${VERL_OPD_DIR}/opd/data/merge_offline_trajectories.py" \
        --mode student \
        --shard-dir "${WORK_ROOT}/student_shards" \
        --jsonl "${STUDENT_JSONL}" \
        --expected-shards "${PHASE_SHARDS}"
else
    echo "[trd-data] skip existing student trajectories: ${STUDENT_JSONL}"
fi

if [[ ! -s "${REWRITE_JSONL}" || ! -s "${REWRITE_JSONL}.summary.json" ]]; then
    echo "[trd-data] phase 2/3: rewrite with teacher"
    run_phase \
        rewrite "${TEACHER_MODEL}" "${STUDENT_JSONL}" "${WORK_ROOT}/rewrite_shards" \
        "${REWRITE_GPU_GROUPS}" "${REWRITE_MAX_MODEL_LEN}" \
        "${REWRITE_MAX_NUM_SEQS}" "${REWRITE_MAX_NUM_BATCHED_TOKENS}"
    "${PYTHON_BIN}" "${VERL_OPD_DIR}/opd/data/merge_offline_trajectories.py" \
        --mode trd \
        --shard-dir "${WORK_ROOT}/rewrite_shards" \
        --jsonl "${REWRITE_JSONL}" \
        --parquet "${OUTPUT_PARQUET}" \
        --expected-shards "${PHASE_SHARDS}"
else
    echo "[trd-data] skip existing rewrites: ${REWRITE_JSONL}"
    if [[ ! -s "${OUTPUT_PARQUET}" ]]; then
        "${PYTHON_BIN}" "${VERL_OPD_DIR}/opd/data/merge_offline_trajectories.py" \
            --mode trd \
            --input-jsonl "${REWRITE_JSONL}" \
            --parquet "${OUTPUT_PARQUET}"
    fi
fi

echo "[trd-data] phase 3/3 complete: ${OUTPUT_PARQUET}"
