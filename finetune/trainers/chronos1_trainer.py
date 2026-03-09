"""
Chronos-1 Trainer using HuggingFace Trainer.

Chronos-1 is a T5-based seq2seq model that tokenizes time series
and trains using cross-entropy loss on the tokenized sequences.

Supports multi-GPU training via torchrun.
"""

import os
import sys
import json
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

# Add chronos-forecasting to path
sys.path.insert(0, "/home/ubuntu/efs/to_customer/zendure/chronos-forecasting/src")

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ..config import TrainingConfig
from ..data.dataset import load_train_files, load_data_from_csv


@dataclass
class ChronosTokenizerConfig:
    """Configuration for Chronos tokenizer."""
    n_tokens: int = 4096
    n_special_tokens: int = 2  # PAD and EOS
    pad_token_id: int = 0
    eos_token_id: int = 1
    context_length: int = 512
    prediction_length: int = 64
    use_eos_token: bool = True


class ChronosTokenizer:
    """
    Tokenizer for Chronos-1 that converts time series to discrete tokens.

    Uses mean-scale uniform binning strategy.
    """

    def __init__(self, config: ChronosTokenizerConfig):
        self.config = config
        self.n_tokens = config.n_tokens
        self.n_special_tokens = config.n_special_tokens
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.context_length = config.context_length
        self.use_eos_token = config.use_eos_token

        # Define bin edges for tokenization
        self._setup_bins()

    def _setup_bins(self):
        """Setup uniform bins for tokenization."""
        # Use uniform bins in [-10, 10] range after scaling
        low = -10.0
        high = 10.0
        n_bins = self.n_tokens - self.n_special_tokens

        self.bin_edges = np.linspace(low, high, n_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def _scale(self, context: np.ndarray) -> tuple:
        """Scale context using mean-scale normalization."""
        # Handle NaN values
        mask = ~np.isnan(context)
        if mask.sum() == 0:
            return context, 0.0, 1.0

        loc = np.nanmean(context)
        scale = np.nanmean(np.abs(context - loc))
        scale = max(scale, 1e-8)

        scaled = (context - loc) / scale
        return scaled, loc, scale

    def encode(
        self,
        context: np.ndarray,
        label: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode time series to token IDs.

        Args:
            context: Input context array
            label: Optional label/target array

        Returns:
            Dictionary with input_ids and (optionally) labels
        """
        # Scale the context
        scaled_context, loc, scale = self._scale(context)

        # Clip to bin range
        clipped = np.clip(scaled_context, self.bin_edges[0], self.bin_edges[-1])

        # Digitize to get bin indices
        token_ids = np.digitize(clipped, self.bin_edges[1:-1])
        token_ids = token_ids + self.n_special_tokens  # Offset by special tokens

        # Handle NaN values (use PAD token)
        nan_mask = np.isnan(context)
        token_ids[nan_mask] = self.pad_token_id

        # Pad or truncate to context_length
        if len(token_ids) > self.context_length:
            token_ids = token_ids[-self.context_length:]
        elif len(token_ids) < self.context_length:
            padding = np.full(
                self.context_length - len(token_ids),
                self.pad_token_id,
                dtype=token_ids.dtype
            )
            token_ids = np.concatenate([padding, token_ids])

        result = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                token_ids != self.pad_token_id, dtype=torch.long
            ),
        }

        if label is not None:
            # Scale labels using same scale as context
            scaled_label = (label - loc) / scale
            clipped_label = np.clip(scaled_label, self.bin_edges[0], self.bin_edges[-1])
            label_ids = np.digitize(clipped_label, self.bin_edges[1:-1])
            label_ids = label_ids + self.n_special_tokens

            # Add EOS token
            if self.use_eos_token:
                label_ids = np.concatenate([label_ids, [self.eos_token_id]])

            result["labels"] = torch.tensor(label_ids, dtype=torch.long)

        return result


class Chronos1Dataset(IterableDataset):
    """
    Iterable dataset for Chronos-1 training.

    Supports distributed training by sharding data across workers.
    """

    def __init__(
        self,
        file_paths: List[str],
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 48,
        min_past: int = 64,
        shuffle_buffer_size: int = 1000,
        seed: int = 42,
    ):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.min_past = min_past
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over samples with pseudo-shuffling."""
        worker_info = torch.utils.data.get_worker_info()

        # Shard files across workers
        if worker_info is not None:
            files = self.file_paths[worker_info.id::worker_info.num_workers]
            seed = self.seed + worker_info.id
        else:
            files = self.file_paths
            seed = self.seed

        rng = np.random.RandomState(seed)
        rng.shuffle(files)

        # Buffer for pseudo-shuffling
        buffer = []

        for file_path in files:
            try:
                samples = self._process_file(file_path)
                buffer.extend(samples)

                # Shuffle and yield when buffer is full
                while len(buffer) >= self.shuffle_buffer_size:
                    rng.shuffle(buffer)
                    for _ in range(self.shuffle_buffer_size // 2):
                        yield buffer.pop()

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        # Yield remaining samples
        rng.shuffle(buffer)
        for sample in buffer:
            yield sample

    def _process_file(self, file_path: str) -> List[Dict[str, torch.Tensor]]:
        """Process a single file and return samples."""
        df = load_data_from_csv(file_path)
        power = df["power"].values.astype(np.float32)

        samples = []
        min_length = self.min_past + self.prediction_length

        if len(power) < min_length:
            return samples

        # Create samples with random context lengths
        rng = np.random.RandomState(hash(file_path) % (2**32))

        for _ in range(10):  # Multiple samples per file
            # Random split point
            split_idx = rng.randint(
                self.min_past,
                len(power) - self.prediction_length + 1
            )

            context = power[:split_idx]
            label = power[split_idx:split_idx + self.prediction_length]

            sample = self.tokenizer.encode(context, label)
            samples.append(sample)

        return samples


class Chronos1Trainer:
    """Trainer for Chronos-1 models using HuggingFace Trainer."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize the Chronos-1 trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load the Chronos-1 model."""
        print(f"Loading Chronos-1 model from {self.config.model_id}")

        # Load model config and model
        model_config = AutoConfig.from_pretrained(self.config.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_id,
            config=model_config,
        )

        # Initialize tokenizer
        tokenizer_config = ChronosTokenizerConfig(
            context_length=self.config.context_length,
            prediction_length=self.config.prediction_length,
        )
        self.tokenizer = ChronosTokenizer(tokenizer_config)

        print(f"Model loaded with {self.model.num_parameters():,} parameters")

    def prepare_data(self) -> tuple:
        """
        Prepare training and validation datasets.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        print("Loading training files...")
        all_train_files = load_train_files(
            self.config.split_csv,
            self.config.data_dir,
            split="train"
        )

        # Split into train/val based on val_ratio
        eval_dataset = None
        if self.config.val_ratio > 0:
            import random
            random.seed(self.config.seed)
            shuffled_files = all_train_files.copy()
            random.shuffle(shuffled_files)

            val_size = int(len(shuffled_files) * self.config.val_ratio)
            val_files = shuffled_files[:val_size]
            train_files = shuffled_files[val_size:]

            print(f"Split {len(all_train_files)} files into {len(train_files)} train, {len(val_files)} val (ratio={self.config.val_ratio})")

            eval_dataset = Chronos1Dataset(
                file_paths=val_files,
                tokenizer=self.tokenizer,
                context_length=self.config.context_length,
                prediction_length=self.config.prediction_length,
                seed=self.config.seed + 1,
            )
        else:
            train_files = all_train_files
            print(f"Using all {len(train_files)} files for training (no validation)")

        train_dataset = Chronos1Dataset(
            file_paths=train_files,
            tokenizer=self.tokenizer,
            context_length=self.config.context_length,
            prediction_length=self.config.prediction_length,
            seed=self.config.seed,
        )

        return train_dataset, eval_dataset

    def train(self) -> None:
        """Run the training loop using HuggingFace Trainer."""
        set_seed(self.config.seed)

        if self.model is None:
            self.load_model()

        train_dataset, eval_dataset = self.prepare_data()

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            max_steps=self.config.num_steps,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            dataloader_num_workers=self.config.num_workers,
            remove_unused_columns=False,
            report_to="none",
            seed=self.config.seed,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self._collate_fn,
        )

        print("Starting Chronos-1 training...")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Num steps: {self.config.num_steps}")

        # Train
        trainer.train()

        # Save final model
        trainer.save_model(os.path.join(self.config.output_dir, "final_model"))
        print(f"Training complete. Model saved to {self.config.output_dir}")

    def _collate_fn(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Collate batch of samples."""
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])

        # Pad labels to same length
        max_label_len = max(b["labels"].size(0) for b in batch)
        labels = torch.full(
            (len(batch), max_label_len),
            -100,  # Ignore index for loss
            dtype=torch.long,
        )
        for i, b in enumerate(batch):
            labels[i, :b["labels"].size(0)] = b["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def save_model(self, output_path: Optional[str] = None) -> None:
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")

        save_path = output_path or os.path.join(self.config.output_dir, "model")
        self.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")


def main():
    """Main function for standalone training."""
    import argparse

    parser = argparse.ArgumentParser(description="Chronos-1 Fine-tuning")
    parser.add_argument("--model_id", type=str, default="amazon/chronos-t5-small")
    parser.add_argument("--data_dir", type=str,
                        default="/home/ubuntu/efs/to_customer/zendure/data/")
    parser.add_argument("--split_csv", type=str,
                        default="/home/ubuntu/efs/to_customer/zendure/zero-shot/data_split.csv")
    parser.add_argument("--output_dir", type=str, default="./output/chronos1/")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    config = TrainingConfig(
        model_type="chronos1",
        model_id=args.model_id,
        data_dir=args.data_dir,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        bf16=args.bf16,
        local_rank=args.local_rank,
    )

    trainer = Chronos1Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
