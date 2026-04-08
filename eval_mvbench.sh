#!/bin/bash
# eval_mvbench.sh
# ===============
# MVBench 评估脚本（短视频，20 个子任务多选题）
#
# 运行位置：video_compress/ 根目录
# 用法：bash eval_mvbench.sh

TASK=mvbench
MODEL_NAME=videochat_flash
MAX_NUM_FRAMES=512

# ── checkpoint 路径（初始验证用 stage2-kf_guided-init，微调后改为 finetune_kf_guided） ──
CKPT_PATH="./checkpoints/stage2-kf_guided-init"

echo "Task      : ${TASK}"
echo "Checkpoint: ${CKPT_PATH}"

TASK_SUFFIX="${TASK//,/_}"
JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
MASTER_PORT=$((18000 + $RANDOM % 100))
NUM_GPUS=8

# lmms_eval 包在当前目录下，通过 PYTHONPATH 确保可导入
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
