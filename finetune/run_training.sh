#!/bin/bash
# Training launch script for Chronos models
# Supports multi-GPU training with different distributed backends

set -e

# Default values
MODEL_TYPE="chronos2"
NUM_GPUS=4
BATCH_SIZE=32
NUM_STEPS=10000
LEARNING_RATE=1e-5
CONTEXT_LENGTH=512
PREDICTION_LENGTH=48
OUTPUT_DIR="./output"
USE_COVARIATES=false
FINETUNE_MODE="full"
VAL_RATIO=0.1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --context_length)
            CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --prediction_length)
            PREDICTION_LENGTH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --use_covariates)
            USE_COVARIATES=true
            shift
            ;;
        --finetune_mode)
            FINETUNE_MODE="$2"
            shift 2
            ;;
        --val_ratio)
            VAL_RATIO="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model_type TYPE       Model type: chronos1, chronos2, chronos_bolt (default: chronos2)"
            echo "  --num_gpus N            Number of GPUs to use (default: 4)"
            echo "  --batch_size N          Batch size per GPU (default: 32)"
            echo "  --num_steps N           Number of training steps (default: 10000)"
            echo "  --learning_rate LR      Learning rate (default: 1e-5)"
            echo "  --context_length N      Context length (default: 512)"
            echo "  --prediction_length N   Prediction length (default: 48)"
            echo "  --output_dir DIR        Output directory (default: ./output)"
            echo "  --use_covariates        Use covariates (Chronos-2 only)"
            echo "  --finetune_mode MODE    Fine-tune mode: full or lora (default: full)"
            echo "  --val_ratio RATIO       Validation ratio (0.0 = no validation, default: 0.1)"
            echo "  --config FILE           Path to YAML config file"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set model ID based on model type
case $MODEL_TYPE in
    chronos1)
        MODEL_ID="amazon/chronos-t5-small"
        ;;
    chronos2)
        MODEL_ID="amazon/chronos-2"
        ;;
    chronos_bolt)
        MODEL_ID="amazon/chronos-bolt-small"
        ;;
    *)
        echo "Unknown model type: $MODEL_TYPE"
        exit 1
        ;;
esac

# Build common arguments
COMMON_ARGS="--model_type $MODEL_TYPE \
    --model_id $MODEL_ID \
    --batch_size $BATCH_SIZE \
    --num_steps $NUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --context_length $CONTEXT_LENGTH \
    --prediction_length $PREDICTION_LENGTH \
    --output_dir ${OUTPUT_DIR}/${MODEL_TYPE}/ \
    --finetune_mode $FINETUNE_MODE \
    --val_ratio $VAL_RATIO"

# Add covariates flag if specified
if [ "$USE_COVARIATES" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --use_covariates"
fi

# Add config file if specified
if [ -n "$CONFIG_FILE" ]; then
    COMMON_ARGS="$COMMON_ARGS --config $CONFIG_FILE"
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"

echo "=============================================="
echo "Chronos Training Launch Script"
echo "=============================================="
echo "Model type:        $MODEL_TYPE"
echo "Model ID:          $MODEL_ID"
echo "Number of GPUs:    $NUM_GPUS"
echo "Batch size:        $BATCH_SIZE"
echo "Number of steps:   $NUM_STEPS"
echo "Learning rate:     $LEARNING_RATE"
echo "Context length:    $CONTEXT_LENGTH"
echo "Prediction length: $PREDICTION_LENGTH"
echo "Output directory:  ${OUTPUT_DIR}/${MODEL_TYPE}/"
echo "Use covariates:    $USE_COVARIATES"
echo "Fine-tune mode:    $FINETUNE_MODE"
echo "Val ratio:         $VAL_RATIO"
echo "=============================================="

# Launch training based on model type
case $MODEL_TYPE in
    chronos1)
        # Chronos-1 uses torchrun for multi-GPU training
        echo "Launching Chronos-1 training with torchrun..."
        torchrun --nproc_per_node=$NUM_GPUS \
            $TRAIN_SCRIPT $COMMON_ARGS
        ;;

    chronos2)
        # Chronos-2 uses accelerate for multi-GPU training
        echo "Launching Chronos-2 training with accelerate..."
        accelerate launch \
            --multi_gpu \
            --num_processes=$NUM_GPUS \
            --mixed_precision=bf16 \
            $TRAIN_SCRIPT $COMMON_ARGS
        ;;

    chronos_bolt)
        # Chronos-Bolt uses accelerate for multi-GPU training
        echo "Launching Chronos-Bolt training with accelerate..."
        accelerate launch \
            --multi_gpu \
            --num_processes=$NUM_GPUS \
            --mixed_precision=bf16 \
            $TRAIN_SCRIPT $COMMON_ARGS
        ;;
esac

echo "Training complete!"
