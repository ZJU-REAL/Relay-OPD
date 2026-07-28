#!/usr/bin/env bash
# Generate one teacher trajectory per prompt for the SFT and SeqKD baselines.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_OPD_DIR=${VERL_OPD_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
TEACHER_MODEL=${TEACHER_MODEL:?TEACHER_MODEL is required}
STUDENT_MODEL=${STUDENT_MODEL:?STUDENT_MODEL is required and supplies the tokenizer/template}
SOURCE_DATA=${SOURCE_DATA:?SOURCE_DATA is required}
OUTPUT_PARQUET=${OUTPUT_PARQUET:?OUTPUT_PARQUET is required}

OUTPUT_JSONL=${OUTPUT_JSONL:-${OUTPUT_PARQUET%.parquet}.jsonl}
WORK_ROOT=${WORK_ROOT:-${OUTPUT_JSONL}.work}
GPU_GROUPS=${GPU_GROUPS:-0;1;2;3;4;5;6;7}
MAX_NEW=${MAX_NEW:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((2048 + MAX_NEW + 1))}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
SEED=${SEED:-0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-65536}
PYTHON_BIN=${PYTHON_BIN:-python3}

IFS=';' read -r -a groups <<< "${GPU_GROUPS}"
num_shards=${#groups[@]}
if ((num_shards == 0)); then
    echo "GPU_GROUPS must contain at least one GPU group" >&2
    exit 2
fi

mkdir -p "${WORK_ROOT}/shards" "${WORK_ROOT}/logs" "$(dirname "${OUTPUT_PARQUET}")"

active_pids=()
cleanup() {
    local pid
    for pid in "${active_pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup INT TERM

echo "[teacher-data] teacher=${TEACHER_MODEL}"
echo "[teacher-data] student tokenizer/template=${STUDENT_MODEL}"
echo "[teacher-data] source=${SOURCE_DATA} output=${OUTPUT_PARQUET}"
echo "[teacher-data] gpu_groups=${GPU_GROUPS} shards=${num_shards} max_new=${MAX_NEW}"

for ((shard = 0; shard < num_shards; shard++)); do
    group=${groups[shard]//[[:space:]]/}
    IFS=',' read -r -a devices <<< "${group}"
    tp=${#devices[@]}
    shard_out="${WORK_ROOT}/shards/shard_${shard}.jsonl"
    if [[ -f "${shard_out}" && -s "${shard_out}.summary.json" ]]; then
        echo "[teacher-data] skip complete shard ${shard}"
        continue
    fi

    (
        export CUDA_VISIBLE_DEVICES="${group}"
        export VLLM_ALLREDUCE_USE_FLASHINFER=0
        export VLLM_USE_FLASHINFER_SAMPLER=0
        export VLLM_ENABLE_V1_MULTIPROCESSING=0
        "${PYTHON_BIN}" "${VERL_OPD_DIR}/opd/data/generate_teacher_trajectories.py" \
            --teacher-model "${TEACHER_MODEL}" \
            --student-tokenizer "${STUDENT_MODEL}" \
            --data "${SOURCE_DATA}" \
            --out "${shard_out}" \
            --shard-id "${shard}" \
            --num-shards "${num_shards}" \
            --max-new "${MAX_NEW}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --temperature "${TEMPERATURE}" \
            --top-p "${TOP_P}" \
            --seed "${SEED}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --tensor-parallel-size "${tp}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    ) >"${WORK_ROOT}/logs/shard_${shard}.log" 2>&1 &
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
    echo "Teacher generation failed; inspect ${WORK_ROOT}/logs" >&2
    exit 1
fi

"${PYTHON_BIN}" "${VERL_OPD_DIR}/opd/data/merge_offline_trajectories.py" \
    --mode teacher \
    --shard-dir "${WORK_ROOT}/shards" \
    --jsonl "${OUTPUT_JSONL}" \
    --parquet "${OUTPUT_PARQUET}" \
    --expected-shards "${num_shards}"

echo "[teacher-data] complete: ${OUTPUT_PARQUET}"
