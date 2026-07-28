# Relay-OPD

This directory contains the training and evaluation code for **Pass the Baton:
Trajectory-Relayed On-Policy Distillation**. The implementation extends verl
with Relay-OPD, the paper baselines, and a vLLM 0.21.0 speculative-decoding
patch used by Relay-OPD and SKD.

## Layout

```text
opd/
  data/                 Offline-trajectory generation and dataset support
  eval/                 Math benchmark evaluation
  patches/vllm/         Relay-OPD and SKD speculative-decoding patch
  reward/               Training-time math reward and graders
  scripts/
    data/               Offline baseline data synthesis
    baselines/          SFT, SeqKD, GRPO, OPD/FastOPD, TRD, and SKD
    relay_opd/           Main Relay-OPD training entry point
    ablations/           One executable script per paper ablation
    evaluation/          Math benchmark evaluation
```

## Paper Configuration

The online methods use the following defaults:

| Setting | Value |
| --- | ---: |
| Max prompt length | 2,048 |
| Max response length | 16,384 |
| Sampling temperature | 1.0 |
| Sampling top-p | 1.0 |
| Rollouts per prompt | 1 (GRPO: 8) |
| Global batch size | 128 |
| PPO mini-batch size | 128 |
| PPO epochs | 1 |
| Learning rate | 1e-6, constant |
| Training epochs | 1 |

Relay-OPD uses handoff top-K `K=5`, at most `M=2` teacher takeovers, and
`L=3` additional teacher paragraphs per takeover. It applies the k1
reverse-KL policy-gradient objective to the actual token in the relay
trajectory on both student and teacher legs.

The default reflection-token bases are `Wait`, `But`, `Hmm`, `Actually`,
`Hold`, `However`, `Yet`, `Oh`, `Alternatively`, `No`, `Ah`, `Oops`, and
`Well`; their case and leading-space variants are resolved with the student
tokenizer. Override the comma-separated set with `RELAY_OPD_REFLECTION_TOKENS`.

All methods use the student's non-thinking Qwen3 template:

```text
<|im_start|>system
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
<think>

</think>
```

## Data

Online methods (`GRPO`, `OPD`, `FastOPD`, `SKD`, and `Relay-OPD`) consume a
verl RL parquet with at least `prompt` and `reward_model` columns.

SFT and SeqKD share one offline dataset containing a single teacher trajectory
per problem. Generate it from the prompt-only parquet with the teacher model,
but render every prompt with the student's tokenizer and non-thinking template:

```bash
TEACHER_MODEL=/path/to/teacher \
STUDENT_MODEL=/path/to/student \
SOURCE_DATA=/path/to/train.parquet \
OUTPUT_PARQUET=/path/to/teacher_trajectories.parquet \
GPU_GROUPS='0;1;2;3;4;5;6;7' \
bash opd/scripts/data/teacher_trajectories.sh
```

Each semicolon-delimited entry in `GPU_GROUPS` is one data-parallel worker;
comma-delimited devices within an entry form its tensor-parallel group. For
example, `GPU_GROUPS='0,1;2,3;4,5;6,7'` runs four TP=2 workers. Generation is
sharded, resumable at shard granularity, and merged only after every shard has
a completion summary.

The resulting SFT/SeqKD parquet contains tokenized `prompt_token_ids` and
`response_token_ids`. Prompt IDs include the empty `<think></think>` block;
only response tokens contribute to SFT loss. SeqKD scores the same response
under the same student-formatted prompt using the teacher.

TRD requires a separate two-stage synthesis pipeline. It first samples one
student trajectory per prompt and then asks the teacher to rewrite that
trajectory. Both stages use the student's tokenizer and non-thinking template:

```bash
STUDENT_MODEL=/path/to/student \
TEACHER_MODEL=/path/to/teacher \
SOURCE_DATA=/path/to/train.parquet \
OUTPUT_PARQUET=/path/to/trd_trajectories.parquet \
STUDENT_GPU_GROUPS='0;1;2;3;4;5;6;7' \
REWRITE_GPU_GROUPS='0,1;2,3;4,5;6,7' \
bash opd/scripts/data/trd_trajectories.sh
```

TRD data additionally contains `teacher_prompt_token_ids`, whose prompt
includes the problem and original student trajectory under the rewrite
instruction. Its student-side `prompt_token_ids` still contain only the
original problem and the empty-thinking assistant prefix.

The TRD rewrite prompt is:

```text
Your task is to rewrite your mathematical solution.
**Problem:** {problem}
**Your Initial Solution:** {initial_response}
**Instructions:**
1. Preserve the overall structure and reasoning path of your original solution
2. Identify and fix errors in computation or logic
3. Keep correct intermediate steps and meaningful work
4. Output ONLY the rewritten solution
```

## Training

Each script takes paths through environment variables and forwards additional
Hydra overrides from its command line. For example:

```bash
export STUDENT_MODEL=/path/to/student
export TEACHER_MODEL=/path/to/teacher
export TRAIN_DATA=/path/to/train.parquet
export BENCH=/path/to/eval_parquets
export OUTPUT_DIR=/path/to/output

bash opd/scripts/relay_opd/train.sh
```

Online distillation defaults to eight GPUs split as four actor/student GPUs and
four teacher-prefill GPUs. A four-GPU 2+2 run can be launched with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACTOR_GPUS_PER_NODE=2 \
TEACHER_GPUS_PER_NODE=2 \
bash opd/scripts/relay_opd/train.sh
```

Offline-data entry points:

| Script | Experiment |
| --- | --- |
| `data/teacher_trajectories.sh` | Shared SFT/SeqKD teacher trajectories |
| `data/trd_trajectories.sh` | Student trajectories and teacher rewrites for TRD |

Baseline entry points:

| Script | Experiment |
| --- | --- |
| `baselines/sft.sh` | SFT on one offline teacher trajectory per prompt |
| `baselines/seqkd.sh` | Offline top-k forward-KL KD |
| `baselines/grpo.sh` | Outcome-reward GRPO |
| `baselines/opd.sh` | Standard 16,384-token PG-style k1 reverse-KL OPD |
| `baselines/fastopd/1024.sh` | FastOPD@1024 |
| `baselines/fastopd/2048.sh` | FastOPD@2048 |
| `baselines/fastopd/4096.sh` | FastOPD@4096 |
| `baselines/fastopd/8192.sh` | FastOPD@8192 |
| `baselines/trd.sh` | TRD rewrite-trajectory KD |
| `baselines/skd.sh` | Speculative Knowledge Distillation |

The main method is launched with `relay_opd/train.sh`.

## Relay-OPD Ablations

Every paper ablation has a dedicated entry point:

| Category | Scripts |
| --- | --- |
| Handoff top-K | `ablations/trigger_topk/k1.sh`, `k10.sh` |
| Relay budget M | `ablations/takeover_count/m1.sh`, `m3.sh`, `m4.sh` |
| Teacher-leg length L | `ablations/takeover_length/l0.sh`, `l1.sh`, `l2.sh`, `l4.sh`, `l5.sh`, `l6.sh` |
| Loss | `ablations/loss/teacher_token_rkl.sh`, `student_action_rkl.sh`, `teacher_fkl.sh` |

For example:

```bash
bash opd/scripts/ablations/loss/teacher_fkl.sh
```

The main `train_relay_opd.sh` entry point uses emitted-token k1 RKL and does not
request top-k distributions. Top-k FKL is retained only for the corresponding
paper ablation.

## Evaluation

`evaluation/math.sh` runs independent data-parallel vLLM shards and aggregates
their JSONL and summary files:

```bash
RUN_NAME=relay_opd \
STEP=35 \
MODEL=/path/to/checkpoint/actor/huggingface \
DATA_DIR=/path/to/eval_parquets \
OUT_ROOT=/path/to/eval_output \
BENCHES=aime24,aime25,aime26,math500,amc23,olympiad,hmmt_feb_2026,hmmt_nov_2025 \
N_SAMPLES=32 MAX_NEW=32768 MAX_MODEL_LEN=34817 DP_SIZE=4 TP=1 \
bash opd/scripts/evaluation/math.sh
```

Each benchmark writes `<bench>.summary.json`; raw generations are retained in
`<bench>.jsonl`. The Python runner `opd/eval/math_benchmarks.py` also supports
direct Relay-OPD, Trigger-Stop, and SKD speculative-rollout evaluation.
