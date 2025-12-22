#!/bin/bash

############### Finetune - Universal GPU Detection ################
export PYTHONPATH=$(pwd)
export BASE_RUN_NAME=resume_finetune_Llava-Onevision-siglip-qwen2.5-stage-4_lora_trajectory

export PREV_STAGE_CHECKPOINT=BAAI/RoboBrain
export VISION_MODEL_VERSION=google/siglip-so400m-patch14-384

export DATA_PATH=/content/drive/MyDrive/BRIN/RoboBrain/scripts/train/yaml/stage_4_trajectory.yaml

export IMAGE_FOLDER=/content/drive/MyDrive/BRIN/RoboBrain/dataset/trajectory/images
export OUTPUT_DIR=/content/drive/MyDrive/BRIN/RoboBrain/checkpoints/${BASE_RUN_NAME}

export PROMPT_VERSION=qwen_2

export IMAGE_ASPECT_RATIO=anyres_max_9
export MM_TUNABLE_PARTS="lora"
export IMAGE_GRID_PINPOINTS="(1x1),...,(6x6)"

export NUM_GPUS=1
export NNODES=1

# Colab-specific: Force offline mode and handle GPU detection safely
export WANDB_MODE=offline

# Detect GPU type and set parameters accordingly
# Use timeout and error handling for Colab environment
GPU_TYPE=$(timeout 10s nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "UNKNOWN")

echo "Detected GPU: $GPU_TYPE"

if [[ "$GPU_TYPE" == *"A100"* ]]; then
    echo "Detected A100 GPU - Using optimized settings"
    export DATA_WORKERS=4
    export DEV_BATCHSIZE=2
    export GRAD_ACC_STEPS=1
    TF32_FLAG="--tf32 True"
    TORCH_COMPILE_FLAG="--torch_compile True --torch_compile_backend inductor"
elif [[ "$GPU_TYPE" == *"T4"* ]]; then
    echo "Detected T4 GPU - Using conservative settings"
    export DATA_WORKERS=2
    export DEV_BATCHSIZE=1
    export GRAD_ACC_STEPS=2
    TF32_FLAG=""
    TORCH_COMPILE_FLAG=""
else
    echo "Detected $GPU_TYPE or no GPU - Using safe default settings"
    export DATA_WORKERS=2
    export DEV_BATCHSIZE=1
    export GRAD_ACC_STEPS=2
    TF32_FLAG=""
    TORCH_COMPILE_FLAG=""
    # Force CPU mode if no GPU detected
    if [[ "$GPU_TYPE" == "UNKNOWN" ]]; then
        echo "No GPU detected, forcing CPU mode"
        export CUDA_VISIBLE_DEVICES=""
        TORCH_COMPILE_FLAG=""
    fi
fi

export LEARNING_RATE=1e-5
export VIT_LEARNING_RATE=2e-6
export MAX_SEQ_LEN=8192
export MAX_FRAME_NUM=8
export ZERO_VERSION=3

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

echo "Starting training with:"
echo "  Batch size: $DEV_BATCHSIZE"
echo "  Gradient accumulation: $GRAD_ACC_STEPS"
echo "  Data workers: $DATA_WORKERS"
echo "  TF32: $([ -n "$TF32_FLAG" ] && echo "Enabled" || echo "Disabled")"
echo "  Torch compile: $([ -n "$TORCH_COMPILE_FLAG" ] && echo "Enabled" || echo "Disabled")"

# Build the command as an array to handle spaces properly
CMD=(
    deepspeed
    --num_gpus "$NUM_GPUS"
    --num_nodes "$NNODES"
    train/train_mem.py
    --deepspeed "scripts/zero${ZERO_VERSION}.json"
    --lora_enable True
    --model_name_or_path "$PREV_STAGE_CHECKPOINT"
    --version "$PROMPT_VERSION"
    --data_path "$DATA_PATH"
    --image_folder "$IMAGE_FOLDER"
    --mm_tunable_parts "$MM_TUNABLE_PARTS"
    --mm_vision_tower_lr "$VIT_LEARNING_RATE"
    --vision_tower "$VISION_MODEL_VERSION"
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --group_by_modality_length True
    --image_aspect_ratio "$IMAGE_ASPECT_RATIO"
    --image_grid_pinpoints "$IMAGE_GRID_PINPOINTS"
    --mm_patch_merge_type spatial_unpad
    --bf16 True
    --run_name "$BASE_RUN_NAME"
    --output_dir "$OUTPUT_DIR"
    --num_train_epochs 1
    --per_device_train_batch_size "$DEV_BATCHSIZE"
    --per_device_eval_batch_size "$DEV_BATCHSIZE"
    --gradient_accumulation_steps "$GRAD_ACC_STEPS"
    --evaluation_strategy "no"
    --save_strategy "steps"
    --save_steps 1000
    --save_total_limit 1
    --learning_rate "$LEARNING_RATE"
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --model_max_length "$MAX_SEQ_LEN"
    --gradient_checkpointing True
    --dataloader_num_workers "$DATA_WORKERS"
    --lazy_preprocess True
    --report_to wandb
    --dataloader_drop_last True
    --frames_upbound "$MAX_FRAME_NUM"
)

# Add optional flags if they exist
if [ -n "$TF32_FLAG" ]; then
    CMD+=(--tf32 True)
fi

# Split TORCH_COMPILE_FLAG into separate arguments
if [ -n "$TORCH_COMPILE_FLAG" ]; then
    CMD+=(--torch_compile True --torch_compile_backend inductor)
fi

echo "Running command:"
echo "${CMD[@]}"

# Execute the command
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"