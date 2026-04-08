#!/bin/bash
# eval_videomme.sh
# ================
# Video-MME 评估脚本（Short / Medium / Long 三段，有/无字幕各一份）
#
# 运行位置：video_compress/ 根目录
# 用法：bash eval_videomme.sh

TASK=videomme,videomme_w_subtitle
MODEL_NAME=videochat_flash
MAX_NUM_FRAMES=512

# ── 修改为微调后的 checkpoint 路径 ────────────────────────────
CKPT_PATH="./checkpoints/finetune_kf_guided/YOUR_RUN_NAME"

echo "Task      : ${TASK}"
echo "Checkpoint: ${CKPT_PATH}"

TASK_SUFFIX="${TASK//,/_}"
JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
MASTER_PORT=$((18000 + $RANDOM % 100))
NUM_GPUS=8

PYTHONPATH=$(pwd):${PYTHONPATH} \
accelerate launch \
    --num_processes ${NUM_GPUS} \
    --main_process_port ${MASTER_PORT} \
    -m lmms_eval \
    --model ${MODEL_NAME} \
    --model_args pretrained=${CKPT_PATH},max_num_frames=${MAX_NUM_FRAMES} \
    --tasks ${TASK} \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${TASK_SUFFIX} \
    --output_path ./logs/${JOB_NAME}_${MODEL_NAME}_f${MAX_NUM_FRAMES}
