#!/usr/bin/env bash
# SFT baseline on one teacher trajectory per prompt.
set -xeuo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_OPD_DIR=${VERL_OPD_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
STUDENT_MODEL=${STUDENT_MODEL:?STUDENT_MODEL is required}
TRAIN_DATA=${TRAIN_DATA:?TRAIN_DATA is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
EXP_ID=${EXP_ID:-$(basename "${OUTPUT_DIR}")}

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/training.log") 2>&1

cd "${VERL_OPD_DIR}"

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}

num_gpus=${NUM_GPUS:-8}
train_batch_size=${TRAIN_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
max_length=${MAX_LENGTH:-$((max_prompt_length + max_response_length + 1))}
max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-24576}
sp_size=${SP_SIZE:-2}
save_freq=${SAVE_FREQ:-5}
total_epochs=${TOTAL_EPOCHS:-1}
checkpoint_save_contents=${CHECKPOINT_SAVE_CONTENTS:-[model,optimizer,extra,hf_model]}

echo "[SFT] experiment=${EXP_ID} output=${OUTPUT_DIR}"
echo "[SFT] student=${STUDENT_MODEL} data=${TRAIN_DATA}"
echo "[SFT] prompt_loss=0 response_loss=1 batch=${train_batch_size} epochs=${total_epochs}"

torchrun --standalone --nnodes=1 --nproc-per-node="${num_gpus}" -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_DATA}" \
    data.val_files=null \
    data.train_batch_size=${train_batch_size} \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=${max_token_len_per_gpu} \
    data.max_length=${max_length} \
    +data.prompt_token_key=prompt_token_ids \
    +data.response_token_key=response_token_ids \
    data.custom_cls.path=${VERL_OPD_DIR}/opd/data/tokenized_sft_dataset.py \
    data.custom_cls.name=TokenizedSFTDataset \
    data.messages_key=messages \
    data.enable_thinking_key=enable_thinking \
    data.enable_thinking_default=False \
    data.ignore_input_ids_mismatch=True \
    data.num_workers=8 \
    model=hf_model \
    model.path="${STUDENT_MODEL}" \
    model.use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    optim=fsdp \
    optim.lr=1e-6 \
    optim.lr_scheduler_type=constant \
    optim.lr_warmup_steps_ratio=0.0 \
    engine=fsdp \
    engine.strategy=fsdp \
    engine.fsdp_size=-1 \
    engine.ulysses_sequence_parallel_size=${sp_size} \
    engine.param_offload=True \
    engine.optimizer_offload=True \
    trainer.logger="['console']" \
    trainer.project_name=sft \
    trainer.experiment_name="${EXP_ID}" \
    trainer.total_epochs=${total_epochs} \
    ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps=${TOTAL_TRAINING_STEPS}} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=-1 \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.resume_mode=disable \
    checkpoint.save_contents="${checkpoint_save_contents}"

echo "=== SFT training complete: ${EXP_ID} ==="
