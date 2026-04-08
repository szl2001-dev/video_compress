#!/bin/bash
# train_kf_guided_subset.sh
# 单卡子集验证：clevrer_qa 20K，16帧，只训练 mm_mlp_adapter
#
# 运行位置：video_compress/ 根目录
# 用法：bash train_kf_guided_subset.sh

export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1

CKPT_PATH="./checkpoints/stage2-kf_guided-init"
DATA_VERSION="data/clevrer_qa_20k.yaml"
PROMPT_VERSION="qwen_2"
WANDB_PROJECT="video_compress"

RUN_NAME=subset_kf_guided_clevrer20k_$(date +"%Y%m%d_%H%M%S")
echo "RUN_NAME: ${RUN_NAME}"

mkdir -p ./output_logs/finetune
mkdir -p ./checkpoints/finetune

PYTHONPATH=$(pwd):${PYTHONPATH} \
WANDB_PROJECT=${WANDB_PROJECT} \
/home/work/miniconda3/envs/video_compress/bin/torchrun \
    --nproc_per_node=1 \
    --master_port=29500 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_VERSION} \
    --vision_tower umt-large \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type kf_guided_mlp \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_nopad \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_nopad \
    --mm_newline_position nothing \
    --bf16 True \
    --run_name ${RUN_NAME} \
    --output_dir ./checkpoints/finetune/${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --dataloader_drop_last True \
    --frames_upbound 16 \
    --frames_lowbound 4 \
    --time_msg short \
    --local_num_frames 4 \
    --vision_encode_type video_image \
    --sample_type dynamic_fps1 \
    --mm_local_num_frames 4 \
    --verbose_logging True \
    2>&1 | tee ./output_logs/finetune/${RUN_NAME}.log
