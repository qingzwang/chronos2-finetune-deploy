#!/usr/bin/env python3
"""
Main training entry point for Chronos models.

Supports Chronos-1, Chronos-2, and Chronos-Bolt with multi-GPU training.

Usage:
    # Chronos-2 with covariates (using Accelerate)
    accelerate launch --multi_gpu --num_processes=4 finetune/train.py \
        --model_type chronos2 \
        --model_id amazon/chronos-t5-small \
        --use_covariates \
        --batch_size 32 \
        --num_steps 10000

    # Chronos-1 (using torchrun)
    torchrun --nproc_per_node=4 finetune/train.py \
        --model_type chronos1 \
        --model_id amazon/chronos-t5-small

    # Chronos-Bolt (using Accelerate)
    accelerate launch --multi_gpu finetune/train.py \
        --model_type chronos_bolt \
        --model_id amazon/chronos-bolt-small
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune.config import TrainingConfig
from finetune.trainers import Chronos1Trainer, Chronos2Trainer, ChronosBoltTrainer


def parse_args() -> Tuple[argparse.Namespace, set]:
    """
    Parse command line arguments.

    Returns:
        Tuple of (args, provided_args) where provided_args is the set of
        argument names explicitly provided on command line.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune Chronos models for time series forecasting",
    )

    # All arguments default to None so we can detect what was explicitly provided
    # Model settings
    parser.add_argument("--model_type", type=str, choices=["chronos1", "chronos2", "chronos_bolt"])
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # Data settings
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--split_csv", type=str)
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--prediction_length", type=int)
    parser.add_argument("--use_covariates", action="store_true", default=None)

    # Training settings
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--finetune_mode", type=str, choices=["full", "lora"])
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--max_grad_norm", type=float)

    # LoRA settings
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)

    # Output settings
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--save_total_limit", type=int)

    # Precision settings
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--no_bf16", action="store_true", default=None)

    # Other settings
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--val_ratio", type=float)
    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()

    # Detect which arguments were explicitly provided on command line
    provided_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            arg_name = arg.split("=")[0].lstrip("-").replace("-", "_")
            provided_args.add(arg_name)

    return args, provided_args


def get_default_model_id(model_type: str) -> str:
    """Get default model ID for a model type."""
    defaults = {
        "chronos1": "amazon/chronos-t5-small",
        "chronos2": "amazon/chronos-2",
        "chronos_bolt": "amazon/chronos-bolt-small",
    }
    return defaults.get(model_type, "amazon/chronos-t5-small")


def main():
    """Main training function."""
    args, provided_args = parse_args()

    # Load config from YAML if provided
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
        # Only override with explicitly provided command line arguments
        for key in provided_args:
            if key == "config":
                continue
            if hasattr(args, key) and hasattr(config, key):
                value = getattr(args, key)
                if value is not None:
                    setattr(config, key, value)

        # Handle special case: --no_bf16 flag
        if "no_bf16" in provided_args and args.no_bf16:
            config.bf16 = False
    else:
        # No config file - model_type is required
        if args.model_type is None:
            raise ValueError("--model_type is required when not using a config file")

        # Use defaults for args not provided
        config = TrainingConfig(
            model_type=args.model_type,
            model_id=args.model_id or get_default_model_id(args.model_type),
            data_dir=args.data_dir or "/home/ubuntu/efs/to_customer/zendure/data/",
            split_csv=args.split_csv or "/home/ubuntu/efs/to_customer/zendure/zero-shot/data_split.csv",
            context_length=args.context_length or 512,
            prediction_length=args.prediction_length or 48,
            use_covariates=args.use_covariates or False,
            batch_size=args.batch_size or 32,
            eval_batch_size=args.eval_batch_size or 64,
            learning_rate=args.learning_rate or 1e-5,
            num_steps=args.num_steps or 10000,
            finetune_mode=args.finetune_mode or "full",
            gradient_accumulation_steps=args.gradient_accumulation_steps or 1,
            warmup_steps=args.warmup_steps or 0,
            weight_decay=args.weight_decay or 0.0,
            max_grad_norm=args.max_grad_norm or 1.0,
            lora_rank=args.lora_rank or 8,
            lora_alpha=args.lora_alpha or 32,
            lora_dropout=args.lora_dropout or 0.1,
            output_dir=args.output_dir or f"./output/{args.model_type}/",
            logging_steps=args.logging_steps or 100,
            save_steps=args.save_steps or 1000,
            eval_steps=args.eval_steps or 500,
            save_total_limit=args.save_total_limit or 3,
            fp16=args.fp16 or False,
            bf16=False if args.no_bf16 else (args.bf16 if args.bf16 is not None else True),
            seed=args.seed or 42,
            num_workers=args.num_workers or 4,
            val_ratio=args.val_ratio if args.val_ratio is not None else 0.1,
            local_rank=args.local_rank or -1,
        )

    # Validate configuration
    config.validate()

    print("=" * 60)
    print("Chronos Fine-tuning Configuration")
    print("=" * 60)
    print(f"Model type:        {config.model_type}")
    print(f"Model ID:          {config.model_id}")
    print(f"Data directory:    {config.data_dir}")
    print(f"Context length:    {config.context_length}")
    print(f"Prediction length: {config.prediction_length}")
    print(f"Use covariates:    {config.use_covariates}")
    print(f"Fine-tune mode:    {config.finetune_mode}")
    print(f"Batch size:        {config.batch_size}")
    print(f"Learning rate:     {config.learning_rate}")
    print(f"Num steps:         {config.num_steps}")
    print(f"Output directory:  {config.output_dir}")
    print(f"BF16:              {config.bf16}")
    print("=" * 60)

    # Select trainer based on model type
    if config.model_type == "chronos1":
        trainer = Chronos1Trainer(config)
    elif config.model_type == "chronos2":
        trainer = Chronos2Trainer(config)
    elif config.model_type == "chronos_bolt":
        trainer = ChronosBoltTrainer(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    # Run training
    trainer.train()

    # Save configuration
    config.to_yaml(os.path.join(config.output_dir, "training_config.yaml"))

    print("Training complete!")


if __name__ == "__main__":
    main()
