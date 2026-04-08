#!/bin/bash
# finetune_kf_guided.sh
# =====================
# 关键帧引导压缩模型的微调脚本
#
# 运行位置：video_compress/ 根目录
#
# 前置步骤：
#   python convert_checkpoint.py \
#       --stage2_path OpenGVLab/stage2-UMT-Qwen2-7B-tome16_mlp \
#       --output_path ./checkpoints/stage2-kf_guided-init
#
# 用法：
#   bash finetune_kf_guided.sh

export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1

# ── 路径配置（根据实际情况修改） ─────────────────────────────
# convert_checkpoint.py 的输出路径
INIT_CKPT="./checkpoints/stage2-kf_guided-init"

# Stage-3 SFT 数据 yaml（需自行准备，格式参考原版 stage3_short-long_mix_sft.yaml）
DATA_VERSION="./data/stage3_short-long_mix_sft.yaml"

# ── 模型 / 训练配置 ───────────────────────────────────────────
VISION_MODEL_VERSION="umt-large"
mm_projector_type="kf_guided_mlp"
PROMPT_VERSION="qwen_2"

DATA_VERSION_CLEAN=$(basename "$DATA_VERSION")
MID_RUN_NAME="finetune-kf_guided-${mm_projector_type}_Qwen2_7B_$(date +"%Y%m%d_%H%M%S")"
echo "RUN_NAME: ${MID_RUN_NAME}"

OUTPUT_DIR="./checkpoints/finetune_kf_guided/${MID_RUN_NAME}"
LOG_DIR="./logs/finetune_kf_guided"
mkdir -p ${LOG_DIR}

# mm_tunable_parts：训练 projector + LLM，冻结视觉编码器（显存不足时可去掉 mm_language_model）
TUNABLE_PARTS="mm_mlp_adapter,mm_language_model"

NUM_GPU=8

# ── Slurm 版本 ────────────────────────────────────────────────
PARTITION='video'
JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --ntasks=${NUM_GPU} \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python -u llava/train/train_mem.py \
    --deepspeed scripts/zero1.json \
    --model_name_or_path ${INIT_CKPT} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_VERSION} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="${TUNABLE_PARTS}" \
    --mm_vision_select_layer -2 \
    --mm_projector_type ${mm_projector_type} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_nopad \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_nopad \
    --mm_newline_position nothing \
    --bf16 True \
    --run_name ${MID_RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 512 \
    --frames_lowbound 64 \
    --time_msg short \
    --local_num_frames 4 \
    --vision_encode_type video_image \
    --sample_type dynamic_fps1 \
    --mm_local_num_frames 4 \
    --verbose_logging True >> ${LOG_DIR}/${MID_RUN_NAME}.log

# ── 非 Slurm / torchrun 版本 ──────────────────────────────────
# 注释掉上面的 srun，取消下面注释：
#
# torchrun \
#     --nproc_per_node=${NUM_GPU} \
#     --master_port=$((12000 + $RANDOM % 1000)) \
#     llava/train/train_mem.py \
#     --deepspeed scripts/zero1.json \
#     --model_name_or_path ${INIT_CKPT} \
#     --version ${PROMPT_VERSION} \
#     --data_path ${DATA_VERSION} \
#     --vision_tower ${VISION_MODEL_VERSION} \
#     --mm_tunable_parts="${TUNABLE_PARTS}" \
#     --mm_vision_select_layer -2 \
#     --mm_projector_type ${mm_projector_type} \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --group_by_modality_length True \
#     --image_aspect_ratio anyres_nopad \
#     --image_grid_pinpoints "(1x1),...,(6x6)" \
#     --mm_patch_merge_type spatial_nopad \
#     --mm_newline_position nothing \
#     --bf16 True \
#     --run_name ${MID_RUN_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 32768 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 6 \
#     --lazy_preprocess True \
#     --report_to tensorboard \
#     --torch_compile True \
#     --torch_compile_backend "inductor" \
#     --dataloader_drop_last True \
#     --frames_upbound 512 \
#     --frames_lowbound 64 \
#     --time_msg short \
#     --local_num_frames 4 \
#     --vision_encode_type video_image \
#     --sample_type dynamic_fps1 \
#     --mm_local_num_frames 4 \
#     --verbose_logging True 2>&1 | tee ${LOG_DIR}/${MID_RUN_NAME}.log
