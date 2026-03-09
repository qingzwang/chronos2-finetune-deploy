"""
Chronos-2 Trainer using the built-in Pipeline.fit() method.

Chronos-2 supports:
- Full fine-tuning or LoRA
- Covariates (temperature, irradiance)
- Multi-GPU training via Accelerate
"""

import os
import sys
from typing import List, Dict, Optional, Union
import torch
import numpy as np

# Add chronos-forecasting to path
sys.path.insert(0, "/home/ubuntu/efs/to_customer/zendure/chronos-forecasting/src")

from ..config import TrainingConfig
from ..data.dataset import load_train_files, load_data_from_csv
from ..data.preprocessing import csv_to_chronos2_input
from ..metrics import compute_metrics, MetricsAccumulator


class Chronos2Trainer:
    """Trainer for Chronos-2 models using the built-in fit() method."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize the Chronos-2 trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> None:
        """Load the Chronos-2 pipeline."""
        from chronos.chronos2 import Chronos2Pipeline

        print(f"Loading Chronos-2 model from {self.config.model_id}")
        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.config.model_id,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
        )
        print(f"Model loaded on {self.device}")

    def prepare_data(self) -> tuple:
        """
        Prepare training and validation data.

        Returns:
            Tuple of (train_inputs, val_inputs)
        """
        print("Loading training files...")
        all_train_files = load_train_files(
            self.config.split_csv,
            self.config.data_dir,
            split="train"
        )

        # Split into train/val based on val_ratio
        val_inputs = None
        if self.config.val_ratio > 0:
            import random
            random.seed(self.config.seed)
            shuffled_files = all_train_files.copy()
            random.shuffle(shuffled_files)

            val_size = int(len(shuffled_files) * self.config.val_ratio)
            val_files = shuffled_files[:val_size]
            train_files = shuffled_files[val_size:]

            print(f"Split {len(all_train_files)} files into {len(train_files)} train, {len(val_files)} val (ratio={self.config.val_ratio})")
            val_inputs = self._convert_files_to_inputs(val_files)
        else:
            train_files = all_train_files
            print(f"Using all {len(train_files)} files for training (no validation)")

        # Convert to Chronos-2 input format
        train_inputs = self._convert_files_to_inputs(train_files)

        return train_inputs, val_inputs

    def _convert_files_to_inputs(
        self,
        file_paths: List[str],
    ) -> List[Union[torch.Tensor, Dict]]:
        """Convert CSV files to Chronos-2 input format."""
        inputs = []

        for file_path in file_paths:
            try:
                df = load_data_from_csv(file_path)

                # Extract target
                target = torch.tensor(df["power"].values, dtype=torch.float32)

                if self.config.use_covariates:
                    # Include covariates
                    # past_covariates: full series (including future values for training)
                    # future_covariates: indicate which covariates are known in future (values=None)
                    temperature = torch.tensor(
                        df["temperature"].values, dtype=torch.float32
                    )
                    irradiance = torch.tensor(
                        df["irradiance"].values, dtype=torch.float32
                    )

                    input_dict = {
                        "target": target,
                        "past_covariates": {
                            "temperature": temperature,
                            "irradiance": irradiance,
                        },
                        # future_covariates with None values indicates these are known-future covariates
                        # Chronos-2 will extract values from past_covariates at the prediction window
                        "future_covariates": {
                            "temperature": None,
                            "irradiance": None,
                        },
                    }
                    inputs.append(input_dict)
                else:
                    # Univariate - just the target tensor
                    inputs.append(target)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        return inputs

    def train(self) -> None:
        """Run the training loop using Chronos-2's built-in fit() method."""
        if self.pipeline is None:
            self.load_model()

        train_inputs, val_inputs = self.prepare_data()

        print(f"Starting Chronos-2 fine-tuning...")
        print(f"  Mode: {self.config.finetune_mode}")
        print(f"  Use covariates: {self.config.use_covariates}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Num steps: {self.config.num_steps}")

        # Prepare LoRA config if needed
        lora_config = None
        if self.config.finetune_mode == "lora":
            from peft import LoraConfig

            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
            )

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Prepare extra trainer kwargs
        extra_kwargs = {
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "lr_scheduler_type": "cosine",  # Cosine annealing instead of linear decay
        }

        # Add warmup if specified
        if self.config.warmup_steps > 0:
            extra_kwargs["warmup_steps"] = self.config.warmup_steps

        # Add data augmentation parameters
        if self.config.augmentation:
            extra_kwargs["augmentation"] = True
            extra_kwargs["aug_jitter_sigma"] = self.config.aug_jitter_sigma
            extra_kwargs["aug_scaling_sigma"] = self.config.aug_scaling_sigma
            extra_kwargs["aug_mag_warp_sigma"] = self.config.aug_mag_warp_sigma
            extra_kwargs["aug_mag_warp_knots"] = self.config.aug_mag_warp_knots
            extra_kwargs["aug_prob"] = self.config.aug_prob
            print(f"  Data augmentation: enabled (jitter={self.config.aug_jitter_sigma}, scale={self.config.aug_scaling_sigma}, mag_warp={self.config.aug_mag_warp_sigma}, knots={self.config.aug_mag_warp_knots}, prob={self.config.aug_prob})")

        # Add covariate dropout
        if self.config.covariate_dropout_prob > 0:
            extra_kwargs["covariate_dropout_prob"] = self.config.covariate_dropout_prob
            print(f"  Covariate dropout: {self.config.covariate_dropout_prob}")

        # Call the built-in fit method
        finetuned_pipeline = self.pipeline.fit(
            inputs=train_inputs,
            prediction_length=self.config.prediction_length,
            validation_inputs=val_inputs,
            finetune_mode=self.config.finetune_mode,
            lora_config=lora_config,
            context_length=self.config.context_length,
            learning_rate=self.config.learning_rate,
            num_steps=self.config.num_steps,
            batch_size=self.config.batch_size,
            output_dir=self.config.output_dir,
            **extra_kwargs,
        )

        print(f"Fine-tuning complete. Model saved to {self.config.output_dir}")

        # Update the pipeline reference
        self.pipeline = finetuned_pipeline

    def save_model(self, output_path: Optional[str] = None) -> None:
        """
        Save the fine-tuned model.

        Args:
            output_path: Optional custom output path
        """
        if self.pipeline is None:
            raise ValueError("No model to save. Train or load a model first.")

        save_path = output_path or os.path.join(self.config.output_dir, "final_model")
        os.makedirs(save_path, exist_ok=True)

        # Save the model
        self.pipeline.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def evaluate(self, test_files: Optional[List[str]] = None) -> Dict:
        """
        Evaluate the model on test data.

        Args:
            test_files: Optional list of test file paths

        Returns:
            Dictionary of evaluation metrics
        """
        if self.pipeline is None:
            raise ValueError("No model to evaluate. Train or load a model first.")

        if test_files is None:
            test_files = load_train_files(
                self.config.split_csv,
                self.config.data_dir,
                split="test"
            )

        if not test_files:
            print("No test files found for evaluation")
            return {}

        print(f"Evaluating on {len(test_files)} test files...")

        metrics_acc = MetricsAccumulator()

        for file_path in test_files[:100]:  # Limit to 100 files for speed
            try:
                df = load_data_from_csv(file_path)

                if len(df) < self.config.context_length + self.config.prediction_length:
                    continue

                # Use the last prediction_length as target
                context_end = len(df) - self.config.prediction_length
                context = torch.tensor(
                    df["power"].values[:context_end], dtype=torch.float32
                )
                target = df["power"].values[context_end:].astype(np.float32)

                # Make prediction
                if self.config.use_covariates:
                    # Include future covariates for prediction
                    future_covariates = {
                        "temperature": torch.tensor(
                            df["temperature"].values[context_end:], dtype=torch.float32
                        ),
                        "irradiance": torch.tensor(
                            df["irradiance"].values[context_end:], dtype=torch.float32
                        ),
                    }
                    forecast = self.pipeline.predict(
                        context=context.unsqueeze(0),
                        prediction_length=self.config.prediction_length,
                        future_covariates=future_covariates,
                    )
                else:
                    forecast = self.pipeline.predict(
                        context=context.unsqueeze(0),
                        prediction_length=self.config.prediction_length,
                    )

                # Get median prediction
                pred = forecast.median(dim=1).values.squeeze().cpu().numpy()

                # Accumulate metrics
                metrics_acc.update(pred, target)

            except Exception as e:
                print(f"Error evaluating {file_path}: {e}")
                continue

        # Compute aggregated metrics
        results = metrics_acc.compute()
        results["num_samples"] = len(metrics_acc._predictions)

        print(f"Evaluation results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

        return results


def main():
    """Main function for standalone training."""
    import argparse

    parser = argparse.ArgumentParser(description="Chronos-2 Fine-tuning")
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2")
    parser.add_argument("--data_dir", type=str,
                        default="/home/ubuntu/efs/to_customer/zendure/data/")
    parser.add_argument("--split_csv", type=str,
                        default="/home/ubuntu/efs/to_customer/zendure/zero-shot/data_split.csv")
    parser.add_argument("--output_dir", type=str, default="./output/chronos2/")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--finetune_mode", type=str, default="full",
                        choices=["full", "lora"])
    parser.add_argument("--use_covariates", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)

    args = parser.parse_args()

    config = TrainingConfig(
        model_type="chronos2",
        model_id=args.model_id,
        data_dir=args.data_dir,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        finetune_mode=args.finetune_mode,
        use_covariates=args.use_covariates,
        bf16=args.bf16,
    )

    trainer = Chronos2Trainer(config)
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
