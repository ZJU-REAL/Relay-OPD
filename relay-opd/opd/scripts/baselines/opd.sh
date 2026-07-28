#!/usr/bin/env bash
# PG-style reverse-KL OPD. Override MAX_RESPONSE_LENGTH to reproduce
# the FastOPD response-length baselines.
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

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/training.log") 2>&1

cd "${VERL_OPD_DIR}"

export MATH_GRADER_PATH=${MATH_GRADER_PATH:-${VERL_OPD_DIR}/opd/reward/grader}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
actor_gpus_per_node=${ACTOR_GPUS_PER_NODE:-4}
teacher_gpus_per_node=${TEACHER_GPUS_PER_NODE:-4}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
val_max_response_length=${VAL_MAX_RESPONSE_LENGTH:-32768}
val_max_model_len=$((max_prompt_length + val_max_response_length + 1))
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-${val_max_model_len}}
teacher_max_model_len=${TEACHER_MAX_MODEL_LEN:-${val_max_model_len}}

rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
teacher_gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION:-0.45}
rollout_tensor_model_parallel_size=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}
teacher_tensor_model_parallel_size=${TEACHER_TENSOR_MODEL_PARALLEL_SIZE:-1}
rollout_moe_backend=${ROLLOUT_MOE_BACKEND:-auto}
teacher_moe_backend=${TEACHER_MOE_BACKEND:-auto}
actor_ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
rollout_log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-24576}
teacher_max_num_batched_tokens=${TEACHER_MAX_NUM_BATCHED_TOKENS:-4096}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:-5}
val_before_train=${VAL_BEFORE_TRAIN:-True}
total_epochs=${TOTAL_EPOCHS:-1}
actor_checkpoint_save_contents=${ACTOR_CHECKPOINT_SAVE_CONTENTS:-[model,optimizer,extra,hf_model]}

echo "[OPD] experiment=${EXP_ID} output=${OUTPUT_DIR}"
echo "[OPD] student=${STUDENT_MODEL} teacher=${TEACHER_MODEL} data=${TRAIN_DATA}"
echo "[OPD] response=${max_response_length} rollout_tp=${rollout_tensor_model_parallel_size} teacher_tp=${teacher_tensor_model_parallel_size}"
echo "[OPD] resources=actor:${actor_gpus_per_node}+teacher:${teacher_gpus_per_node} GPUs"

python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tensor_model_parallel_size} \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=${rollout_moe_backend} \
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
    trainer.project_name=opd \
    trainer.experiment_name=${EXP_ID} \
    trainer.n_gpus_per_node=${actor_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
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
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tensor_model_parallel_size} \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_memory_utilization} \
    distillation.teacher_models.teacher_model.inference.max_model_len=${teacher_max_model_len} \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${teacher_max_num_batched_tokens} \
    ++distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.moe_backend=${teacher_moe_backend} \
    distillation.distillation_loss.loss_mode=k1 \
    distillation.distillation_loss.use_policy_gradient=True \
    distillation.distillation_loss.use_task_rewards=False \
    "$@"

echo "=== OPD training complete: ${EXP_ID} ==="
