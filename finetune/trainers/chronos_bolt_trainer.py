"""
Chronos-Bolt Trainer with custom training loop.

Chronos-Bolt uses quantile regression loss and supports
multi-GPU training via Accelerate.
"""

import os
import sys
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

# Add chronos-forecasting to path
sys.path.insert(0, "/home/ubuntu/efs/to_customer/zendure/chronos-forecasting/src")

from ..config import TrainingConfig
from ..data.dataset import load_train_files, load_data_from_csv, ChronosFinetuneDataset
from ..metrics import MetricsAccumulator


class QuantileLoss(nn.Module):
    """Quantile (Pinball) Loss for probabilistic forecasting."""

    def __init__(self, quantiles: List[float] = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.quantiles = torch.tensor(quantiles)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute quantile loss.

        Args:
            predictions: Shape (batch, num_quantiles, prediction_length)
            targets: Shape (batch, prediction_length)
            mask: Optional mask for valid positions

        Returns:
            Scalar loss value
        """
        # Move quantiles to same device
        quantiles = self.quantiles.to(predictions.device)

        # Expand targets for quantile comparison
        # Shape: (batch, num_quantiles, prediction_length)
        targets_expanded = targets.unsqueeze(1).expand_as(predictions)

        # Quantile loss: 2 * |q - 1(y <= y_hat)| * |y - y_hat|
        errors = targets_expanded - predictions
        indicator = (errors <= 0).float()

        # Shape: (batch, num_quantiles, prediction_length)
        loss = 2 * torch.abs(quantiles.view(1, -1, 1) - indicator) * torch.abs(errors)

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(loss)
            loss = loss * mask.float()
            loss = loss.sum() / (mask.sum() * len(quantiles) + 1e-8)
        else:
            loss = loss.mean()

        return loss


class ChronosBoltTrainer:
    """Trainer for Chronos-Bolt models with custom training loop."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize the Chronos-Bolt trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> None:
        """Load the Chronos-Bolt model."""
        from chronos import ChronosBoltPipeline

        print(f"Loading Chronos-Bolt model from {self.config.model_id}")
        self.pipeline = ChronosBoltPipeline.from_pretrained(
            self.config.model_id,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
        )
        self.model = self.pipeline.model
        print(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def prepare_data(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare training and validation dataloaders.

        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        print("Loading training files...")
        all_train_files = load_train_files(
            self.config.split_csv,
            self.config.data_dir,
            split="train"
        )

        # Split into train/val based on val_ratio
        val_loader = None
        if self.config.val_ratio > 0:
            import random
            random.seed(self.config.seed)
            shuffled_files = all_train_files.copy()
            random.shuffle(shuffled_files)

            val_size = int(len(shuffled_files) * self.config.val_ratio)
            val_files = shuffled_files[:val_size]
            train_files = shuffled_files[val_size:]

            print(f"Split {len(all_train_files)} files into {len(train_files)} train, {len(val_files)} val (ratio={self.config.val_ratio})")

            val_dataset = ChronosFinetuneDataset(
                file_paths=val_files,
                context_length=self.config.context_length,
                prediction_length=self.config.prediction_length,
                use_covariates=False,
                stride=self.config.prediction_length,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
        else:
            train_files = all_train_files
            print(f"Using all {len(train_files)} files for training (no validation)")

        # Create training dataset
        train_dataset = ChronosFinetuneDataset(
            file_paths=train_files,
            context_length=self.config.context_length,
            prediction_length=self.config.prediction_length,
            use_covariates=False,  # Bolt doesn't support covariates
            stride=self.config.prediction_length,  # Non-overlapping windows
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        return train_loader, val_loader

    def _normalize_batch(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply instance normalization to batch.

        Args:
            context: Shape (batch, context_length)
            target: Shape (batch, prediction_length)

        Returns:
            Tuple of (normalized_context, normalized_target, loc, scale)
        """
        # Compute mean and scale from context
        loc = torch.nanmean(context, dim=-1, keepdim=True)
        scale = torch.sqrt(torch.nanmean((context - loc) ** 2, dim=-1, keepdim=True))
        scale = torch.clamp(scale, min=1e-8)

        # Normalize
        context_norm = (context - loc) / scale
        target_norm = (target - loc) / scale

        return context_norm, target_norm, loc.squeeze(-1), scale.squeeze(-1)

    def _denormalize(
        self,
        predictions: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denormalize predictions.

        Args:
            predictions: Shape (batch, num_quantiles, prediction_length)
            loc: Shape (batch,)
            scale: Shape (batch,)

        Returns:
            Denormalized predictions
        """
        loc = loc.unsqueeze(-1).unsqueeze(-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        return predictions * scale + loc

    def train(self) -> None:
        """Run the training loop."""
        try:
            from accelerate import Accelerator
            use_accelerate = True
        except ImportError:
            use_accelerate = False
            print("Accelerate not available, using single GPU training")

        if self.model is None:
            self.load_model()

        train_loader, val_loader = self.prepare_data()

        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.config.num_steps,
        )

        loss_fn = QuantileLoss()

        # Setup accelerator for multi-GPU
        if use_accelerate:
            accelerator = Accelerator(
                mixed_precision="bf16" if self.config.bf16 else "no",
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            )
            self.model, optimizer, train_loader, scheduler = accelerator.prepare(
                self.model, optimizer, train_loader, scheduler
            )
            if val_loader:
                val_loader = accelerator.prepare(val_loader)
            device = accelerator.device
            is_main = accelerator.is_main_process
        else:
            self.model = self.model.to(self.device)
            device = self.device
            is_main = True

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        print("Starting Chronos-Bolt training...")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Num steps: {self.config.num_steps}")

        # Training loop
        self.model.train()
        global_step = 0
        running_loss = 0.0

        train_iter = iter(train_loader)

        while global_step < self.config.num_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            context = batch["target_context"].to(device)
            target = batch["target_future"].to(device)

            # Normalize
            context_norm, target_norm, loc, scale = self._normalize_batch(context, target)

            # Forward pass
            # Chronos-Bolt expects context in shape (batch, history_length)
            predictions = self.model(context_norm, prediction_length=self.config.prediction_length)

            # Compute loss on normalized values
            loss = loss_fn(predictions, target_norm)

            # Backward pass
            if use_accelerate:
                accelerator.backward(loss)
            else:
                loss.backward()

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                if use_accelerate:
                    accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            global_step += 1

            # Logging
            if global_step % self.config.logging_steps == 0 and is_main:
                avg_loss = running_loss / self.config.logging_steps
                lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step}/{self.config.num_steps} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                running_loss = 0.0

            # Evaluation
            if val_loader and global_step % self.config.eval_steps == 0:
                val_metrics = self._evaluate(val_loader, loss_fn, device)
                if is_main:
                    metrics_str = " | ".join(
                        f"{k}: {v:.4f}" for k, v in val_metrics.items()
                    )
                    print(f"  Validation: {metrics_str}")
                self.model.train()

            # Save checkpoint
            if global_step % self.config.save_steps == 0 and is_main:
                self._save_checkpoint(global_step)

        # Save final model
        if is_main:
            self._save_checkpoint("final")
            print(f"Training complete. Model saved to {self.config.output_dir}")

    def _evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate on validation set and compute metrics."""
        self.model.eval()
        metrics_acc = MetricsAccumulator()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                context = batch["target_context"].to(device)
                target = batch["target_future"].to(device)

                context_norm, target_norm, loc, scale = self._normalize_batch(context, target)
                predictions = self.model(context_norm, prediction_length=self.config.prediction_length)
                loss = loss_fn(predictions, target_norm)

                # Get median prediction (index 4 for 9 quantiles) and denormalize
                median_pred = predictions[:, 4, :]  # Shape: (batch, prediction_length)
                median_pred_denorm = self._denormalize(
                    median_pred.unsqueeze(1), loc, scale
                ).squeeze(1)

                # Accumulate metrics on denormalized values
                metrics_acc.update(median_pred_denorm, target, loss.item())

                if batch_idx >= 50:  # Limit eval batches
                    break

        return metrics_acc.compute()

    def _save_checkpoint(self, step: str) -> None:
        """Save model checkpoint."""
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)

        # Save model state
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(save_path)
        print(f"Checkpoint saved to {save_path}")

    def save_model(self, output_path: Optional[str] = None) -> None:
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")

        save_path = output_path or os.path.join(self.config.output_dir, "model")
        os.makedirs(save_path, exist_ok=True)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(save_path)
        print(f"Model saved to {save_path}")


def main():
    """Main function for standalone training."""
    import argparse

    parser = argparse.ArgumentParser(description="Chronos-Bolt Fine-tuning")
    parser.add_argument("--model_id", type=str, default="amazon/chronos-bolt-small")
    parser.add_argument("--data_dir", type=str,
                        default="/home/ubuntu/efs/to_customer/zendure/data/")
    parser.add_argument("--split_csv", type=str,
                        default="/home/ubuntu/efs/to_customer/zendure/zero-shot/data_split.csv")
    parser.add_argument("--output_dir", type=str, default="./output/chronos_bolt/")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--bf16", action="store_true", default=True)

    args = parser.parse_args()

    config = TrainingConfig(
        model_type="chronos_bolt",
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
    )

    trainer = ChronosBoltTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
