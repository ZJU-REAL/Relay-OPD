#!/usr/bin/env bash
# Speculative Knowledge Distillation (SKD) baseline.
set -xeuo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_OPD_DIR=${VERL_OPD_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
STUDENT_MODEL=${STUDENT_MODEL:?STUDENT_MODEL is required}
TEACHER_MODEL=${TEACHER_MODEL:?TEACHER_MODEL is required}
TRAIN_DATA=${TRAIN_DATA:?TRAIN_DATA is required}
BENCH=${BENCH:?BENCH is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
EXP_ID=${EXP_ID:-$(basename "${OUTPUT_DIR}")}

LOSS_MODE=${LOSS_MODE:-forward_kl_topk_teacher_renorm}
DISTILL_TOPK=${DISTILL_TOPK:-128}
SKD_ACCEPTANCE_TOPK=${SKD_ACCEPTANCE_TOPK:-5}
export SKD_ACCEPTANCE_TOPK

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/training.log") 2>&1
cd "${VERL_OPD_DIR}"

export MATH_GRADER_PATH=${MATH_GRADER_PATH:-${VERL_OPD_DIR}/opd/reward/grader}
export PYTHONPATH=${VERL_OPD_DIR}/opd/patches/vllm:${PYTHONPATH:-}
export VERL_OPD_VLLM_PATCH=1
export VERL_OPD_VLLM_PATCH_DEFER=1
export VERL_OPD_ROLLOUT_MODE=skd
export VERL_OPD_DRAFT_SAMPLING=1
export VERL_OPD_COLLECT_DRAFT_PROBS=${VERL_OPD_COLLECT_DRAFT_PROBS:-0}
export VERL_OPD_TOKENIZER_PATH=${STUDENT_MODEL}
export VERL_OPD_SOURCE_METRICS=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VERL_OPD_SPECULATIVE_ENABLE=1
export VERL_OPD_TARGET_MODEL=${TEACHER_MODEL}
export VERL_OPD_DRAFT_MODEL=${STUDENT_MODEL}
export VERL_OPD_NUM_SPECULATIVE_TOKENS=${VERL_OPD_NUM_SPECULATIVE_TOKENS:-4}
export VERL_OPD_DRAFT_TENSOR_PARALLEL_SIZE=${VERL_OPD_DRAFT_TENSOR_PARALLEL_SIZE:-1}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export VLLM_ALLREDUCE_USE_FLASHINFER=${VLLM_ALLREDUCE_USE_FLASHINFER:-0}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
actor_gpus_per_node=${ACTOR_GPUS_PER_NODE:-4}
teacher_gpus_per_node=${TEACHER_GPUS_PER_NODE:-4}
grad_accum_steps=${GRAD_ACCUM_STEPS:-2}
actor_sp=${ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
val_max_response_length=${VAL_MAX_RESPONSE_LENGTH:-32768}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length + 1))}
teacher_max_model_len=${TEACHER_MAX_MODEL_LEN:-$((max_prompt_length + val_max_response_length + 1))}
rollout_tp=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}
teacher_tp=${TEACHER_TENSOR_MODEL_PARALLEL_SIZE:-1}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
teacher_gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION:-0.45}
actor_ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-9600}
rollout_log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-24576}
teacher_max_num_batched_tokens=${TEACHER_MAX_NUM_BATCHED_TOKENS:-4096}
actor_checkpoint_save_contents=${ACTOR_CHECKPOINT_SAVE_CONTENTS:-[model,optimizer,extra,hf_model]}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:--1}
val_before_train=${VAL_BEFORE_TRAIN:-False}
total_epochs=${TOTAL_EPOCHS:-1}

echo "[SKD] experiment=${EXP_ID} student=${STUDENT_MODEL} teacher=${TEACHER_MODEL}"
echo "[SKD] acceptance_topk=${SKD_ACCEPTANCE_TOPK} speculative_tokens=${VERL_OPD_NUM_SPECULATIVE_TOKENS}"
echo "[SKD] loss=${LOSS_MODE} distill_topk=${DISTILL_TOPK} response=${max_response_length}"
echo "[SKD] resources=actor:${actor_gpus_per_node}+teacher:${teacher_gpus_per_node} GPUs"

python3 -m verl.trainer.main_ppo \
    hydra.run.dir="${OUTPUT_DIR}/hydra" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['${TRAIN_DATA}']" \
    "data.val_files=['${BENCH}/aime-24_verl.parquet','${BENCH}/aime-2025_verl.parquet']" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    data.seed=42 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.gradient_accumulation_steps=${grad_accum_steps} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${actor_sp} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    ++actor_rollout_ref.rollout.val_kwargs.n=32 \
    ++actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    ++actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    ++actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    ++actor_rollout_ref.rollout.val_kwargs.max_tokens=${val_max_response_length} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${rollout_log_prob_max_token_len_per_gpu} \
    trainer.logger=console \
    trainer.project_name=skd \
    trainer.experiment_name=${EXP_ID} \
    trainer.n_gpus_per_node=${actor_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps=${TOTAL_TRAINING_STEPS}} \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    ++trainer.validation_data_dir="${OUTPUT_DIR}/val_generations" \
    +actor_rollout_ref.actor.checkpoint.save_contents="${actor_checkpoint_save_contents}" \
    reward.reward_manager.source=register \
    reward.reward_manager.name=remote \
    reward.num_workers=4 \
    reward.custom_reward_function.path=${VERL_OPD_DIR}/opd/reward/math_reward.py \
    reward.custom_reward_function.name=compute_score \
    distillation.enabled=True \
    distillation.n_gpus_per_node=${teacher_gpus_per_node} \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp} \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_memory_utilization} \
    distillation.teacher_models.teacher_model.inference.max_model_len=${teacher_max_model_len} \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${teacher_max_num_batched_tokens} \
    distillation.distillation_loss.loss_mode=${LOSS_MODE} \
    distillation.distillation_loss.topk=${DISTILL_TOPK} \
    distillation.distillation_loss.use_policy_gradient=False \
    distillation.distillation_loss.use_task_rewards=False \
    "$@"

echo "=== SKD training complete: ${EXP_ID} ==="
