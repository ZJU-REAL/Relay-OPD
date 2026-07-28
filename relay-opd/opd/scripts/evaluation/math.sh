#!/bin/bash
# Multi-GPU math evaluation using independent data-parallel vLLM shards.
set -euo pipefail

export VLLM_ALLREDUCE_USE_FLASHINFER=${VLLM_ALLREDUCE_USE_FLASHINFER:-0}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_OPD_DIR=${VERL_OPD_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
EVAL_SCRIPT=${EVAL_SCRIPT:-${VERL_OPD_DIR}/opd/eval/math_benchmarks.py}
DATA_DIR=${DATA_DIR:?DATA_DIR is required}
MATH_GRADER_PATH=${MATH_GRADER_PATH:-${VERL_OPD_DIR}/opd/reward/grader}
export MATH_GRADER_PATH

RUN_NAME=${RUN_NAME:?set RUN_NAME}
STEP=${STEP:?set STEP}
MODEL=${MODEL:?set MODEL}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT}

BENCHES=${BENCHES:-aime24,aime25}
N_SAMPLES=${N_SAMPLES:-32}
MAX_NEW=${MAX_NEW:-32768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-34817}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
GPU_MEM=${GPU_MEM:-0.90}
TP=${TP:-1}
DP_SIZE=${DP_SIZE:-4}
SEED=${SEED:-42}

OUT_DIR="${OUT_ROOT}/${RUN_NAME}/step_${STEP}"
LOG_DIR="${OUT_ROOT}/job_logs/${RUN_NAME}_step_${STEP}"
RESULTS="${OUT_ROOT}/results.tsv"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "=== Math benchmark evaluation ==="
echo "run_name=${RUN_NAME}"
echo "step=${STEP}"
echo "model=${MODEL}"
echo "out_dir=${OUT_DIR}"
echo "eval_script=${EVAL_SCRIPT}"
echo "data_dir=${DATA_DIR}"
echo "grader_path=${MATH_GRADER_PATH}"
echo "params: benches=${BENCHES} n_samples=${N_SAMPLES} max_new=${MAX_NEW} max_model_len=${MAX_MODEL_LEN} temp=${TEMPERATURE} top_p=${TOP_P} gpu_mem=${GPU_MEM} tp=${TP} dp_size=${DP_SIZE} seed=${SEED}"

if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "missing model config: ${MODEL}/config.json" >&2
  exit 1
fi

if (( DP_SIZE * TP > 8 )); then
  echo "invalid eval parallelism: DP_SIZE * TP must be <= 8, got ${DP_SIZE} * ${TP}" >&2
  exit 2
fi

IFS=',' read -r -a BENCH_ARRAY <<< "${BENCHES}"
if [[ -z "${BENCH_N_SAMPLES:-}" ]]; then
  BENCH_N_SAMPLES=""
  for _bench in "${BENCH_ARRAY[@]}"; do
    if [[ -z "${BENCH_N_SAMPLES}" ]]; then
      BENCH_N_SAMPLES="${N_SAMPLES}"
    else
      BENCH_N_SAMPLES="${BENCH_N_SAMPLES},${N_SAMPLES}"
    fi
  done
fi

all_done=1
for bench in "${BENCH_ARRAY[@]}"; do
  if [[ ! -f "${OUT_DIR}/${bench}.summary.json" ]]; then
    all_done=0
  fi
done

if [[ "${all_done}" == "1" ]]; then
  echo "[skip] aggregate summaries already exist in ${OUT_DIR}: ${BENCHES}"
else
  cd "${VERL_OPD_DIR}"
  pids=()
  for shard_id in $(seq 0 $((DP_SIZE - 1))); do
    (
      set -euo pipefail
      gpu_start=$((shard_id * TP))
      gpu_list=""
      for off in $(seq 0 $((TP - 1))); do
        gpu=$((gpu_start + off))
        if [[ -z "${gpu_list}" ]]; then
          gpu_list="${gpu}"
        else
          gpu_list="${gpu_list},${gpu}"
        fi
      done
      export CUDA_VISIBLE_DEVICES="${gpu_list}"
      shard_out="${OUT_DIR}/shard_${shard_id}"
      mkdir -p "${shard_out}"
      echo "[shard ${shard_id}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} out=${shard_out}"
      python3 "${EVAL_SCRIPT}" \
        --model "${MODEL}" \
        --benches "${BENCHES}" \
        --bench_n_samples "${BENCH_N_SAMPLES}" \
        --data_dir "${DATA_DIR}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --max_new "${MAX_NEW}" \
        --max_model_len "${MAX_MODEL_LEN}" \
        --gpu_mem "${GPU_MEM}" \
        --tp "${TP}" \
        --seed "${SEED}" \
        --disable_thinking \
        --num_shards "${DP_SIZE}" \
        --shard_id "${shard_id}" \
        --out_dir "${shard_out}"
    ) >"${LOG_DIR}/shard_${shard_id}.log" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "one or more eval shards failed; shard logs:" >&2
    ls -l "${LOG_DIR}" >&2 || true
    exit 1
  fi
fi

python3 - "${RUN_NAME}" "${STEP}" "${MODEL}" "${OUT_DIR}" "${RESULTS}" "${BENCHES}" "${DP_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

run_name, step, model, out_dir, results, benches, dp_size = sys.argv[1:]
out_dir = Path(out_dir)
results = Path(results)
dp_size = int(dp_size)
results.parent.mkdir(parents=True, exist_ok=True)

if not results.exists():
    results.write_text(
        "run\tstep\tbench\tavg@32\tpass@1\tpass@32\tn_problems\twall_seconds\tmax_new\tmax_model_len\tmodel\n",
        encoding="utf-8",
    )

with results.open("a", encoding="utf-8") as f:
    for bench in [b.strip() for b in benches.split(",") if b.strip()]:
        summaries = []
        combined_jsonl = out_dir / f"{bench}.jsonl"
        with combined_jsonl.open("w", encoding="utf-8") as out_f:
            for shard_id in range(dp_size):
                shard_dir = out_dir / f"shard_{shard_id}"
                summary_path = shard_dir / f"{bench}.summary.json"
                if not summary_path.exists():
                    raise FileNotFoundError(summary_path)
                d = json.loads(summary_path.read_text(encoding="utf-8"))
                summaries.append(d)
                jsonl_path = shard_dir / f"{bench}.jsonl"
                if jsonl_path.exists():
                    with jsonl_path.open("r", encoding="utf-8") as in_f:
                        for line in in_f:
                            if not line.strip():
                                continue
                            rec = json.loads(line)
                            rec["eval_shard_id"] = shard_id
                            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        total_n = sum(int(d.get("n_problems", 0)) for d in summaries)
        if total_n <= 0:
            raise ValueError(f"no problems aggregated for {bench}")
        avg_at_k = sum(float(d.get("avg@k", 0.0)) * int(d.get("n_problems", 0)) for d in summaries) / total_n
        pass_at_1 = sum(float(d.get("pass@1", 0.0)) * int(d.get("n_problems", 0)) for d in summaries) / total_n
        pass_at_k = sum(float(d.get("pass@k", 0.0)) * int(d.get("n_problems", 0)) for d in summaries) / total_n
        wall_seconds = max(float(d.get("wall_seconds", 0.0)) for d in summaries)
        settings = dict(summaries[0].get("settings", {}))
        settings["dp_size"] = dp_size
        settings["tp_per_shard"] = settings.get("tp", 1)
        settings["num_shards"] = dp_size
        aggregate = {
            "tag": bench,
            "model": model,
            "bench": bench,
            "n_problems": total_n,
            "n_samples": int(summaries[0].get("n_samples", 0)),
            "avg@k": avg_at_k,
            "pass@1": pass_at_1,
            "pass@k": pass_at_k,
            "wall_seconds": wall_seconds,
            "settings": settings,
            "shards": summaries,
        }
        (out_dir / f"{bench}.summary.json").write_text(
            json.dumps(aggregate, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        f.write(
            f"{run_name}\t{step}\t{bench}\t"
            f"{avg_at_k:.6f}\t{pass_at_1:.6f}\t{pass_at_k:.6f}\t"
            f"{total_n}\t{wall_seconds:.3f}\t"
            f"{settings.get('max_new', '')}\t{settings.get('max_model_len', '')}\t"
            f"{model}\n"
        )

print(f"[aggregate] wrote {out_dir} and {results}")
PY

echo "=== finished ${RUN_NAME} step ${STEP} ==="
