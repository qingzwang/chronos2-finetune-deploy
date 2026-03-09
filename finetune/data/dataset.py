"""
Dataset classes for Chronos fine-tuning.
"""

import os
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def load_train_files(
    split_csv: str,
    data_dir: str,
    split: str = "train"
) -> List[str]:
    """
    Load file paths for a specific split from the data_split.csv file.

    Args:
        split_csv: Path to data_split.csv file
        data_dir: Directory containing the CSV data files
        split: One of "train", "val", or "test"

    Returns:
        List of full file paths for the specified split
    """
    df = pd.read_csv(split_csv)
    split_files = df[df["split"] == split]["filename"].tolist()
    file_paths = [os.path.join(data_dir, f) for f in split_files]

    # Filter out non-existent files
    existing_files = [f for f in file_paths if os.path.exists(f)]
    if len(existing_files) < len(file_paths):
        missing = len(file_paths) - len(existing_files)
        print(f"Warning: {missing} files not found in {data_dir}")

    return existing_files


def load_data_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load data from a single CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with columns: date, power, temperature, irradiance
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


class ChronosFinetuneDataset(Dataset):
    """
    PyTorch Dataset for Chronos fine-tuning.

    Supports both univariate (power only) and multivariate (with covariates) data.
    """

    def __init__(
        self,
        file_paths: List[str],
        context_length: int = 512,
        prediction_length: int = 48,
        use_covariates: bool = False,
        stride: int = 1,
        min_length: Optional[int] = None,
    ):
        """
        Initialize the dataset.

        Args:
            file_paths: List of CSV file paths
            context_length: Number of time steps for context
            prediction_length: Number of time steps to predict
            use_covariates: Whether to include temperature and irradiance
            stride: Stride for sliding window
            min_length: Minimum series length to include
        """
        self.file_paths = file_paths
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.use_covariates = use_covariates
        self.stride = stride
        self.min_length = min_length or (context_length + prediction_length)

        # Load and preprocess all data
        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> List[Dict]:
        """Load all CSV files and prepare training samples."""
        samples = []

        for file_path in self.file_paths:
            try:
                df = load_data_from_csv(file_path)
                if len(df) < self.min_length:
                    continue

                # Extract time series data
                power = df["power"].values.astype(np.float32)
                temperature = df["temperature"].values.astype(np.float32)
                irradiance = df["irradiance"].values.astype(np.float32)

                # Create sliding window samples
                total_length = self.context_length + self.prediction_length
                for start in range(0, len(df) - total_length + 1, self.stride):
                    end = start + total_length
                    context_end = start + self.context_length

                    sample = {
                        "target_context": power[start:context_end],
                        "target_future": power[context_end:end],
                        "file_id": Path(file_path).stem,
                    }

                    if self.use_covariates:
                        sample["temperature_context"] = temperature[start:context_end]
                        sample["temperature_future"] = temperature[context_end:end]
                        sample["irradiance_context"] = irradiance[start:context_end]
                        sample["irradiance_future"] = irradiance[context_end:end]

                    samples.append(sample)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        print(f"Prepared {len(samples)} training samples from {len(self.file_paths)} files")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        result = {
            "target_context": torch.tensor(sample["target_context"], dtype=torch.float32),
            "target_future": torch.tensor(sample["target_future"], dtype=torch.float32),
        }

        if self.use_covariates:
            result["temperature_context"] = torch.tensor(
                sample["temperature_context"], dtype=torch.float32
            )
            result["temperature_future"] = torch.tensor(
                sample["temperature_future"], dtype=torch.float32
            )
            result["irradiance_context"] = torch.tensor(
                sample["irradiance_context"], dtype=torch.float32
            )
            result["irradiance_future"] = torch.tensor(
                sample["irradiance_future"], dtype=torch.float32
            )

        return result


class ChronosStreamingDataset(Dataset):
    """
    Memory-efficient streaming dataset that loads files on demand.

    Useful when the full dataset doesn't fit in memory.
    """

    def __init__(
        self,
        file_paths: List[str],
        context_length: int = 512,
        prediction_length: int = 48,
        use_covariates: bool = False,
        samples_per_file: int = 10,
    ):
        """
        Initialize the streaming dataset.

        Args:
            file_paths: List of CSV file paths
            context_length: Number of time steps for context
            prediction_length: Number of time steps to predict
            use_covariates: Whether to include covariates
            samples_per_file: Number of random samples per file per epoch
        """
        self.file_paths = file_paths
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.use_covariates = use_covariates
        self.samples_per_file = samples_per_file
        self.total_length = context_length + prediction_length

        # Pre-compute valid files (those with enough data)
        self._validate_files()

    def _validate_files(self):
        """Check which files have enough data."""
        self.valid_files = []
        for file_path in self.file_paths:
            try:
                df = pd.read_csv(file_path)
                if len(df) >= self.total_length:
                    self.valid_files.append(file_path)
            except Exception:
                continue
        print(f"Found {len(self.valid_files)} valid files out of {len(self.file_paths)}")

    def __len__(self) -> int:
        return len(self.valid_files) * self.samples_per_file

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx = idx // self.samples_per_file
        file_path = self.valid_files[file_idx]

        df = load_data_from_csv(file_path)

        # Random start position
        max_start = len(df) - self.total_length
        start = np.random.randint(0, max_start + 1)
        context_end = start + self.context_length
        end = start + self.total_length

        power = df["power"].values.astype(np.float32)

        result = {
            "target_context": torch.tensor(power[start:context_end], dtype=torch.float32),
            "target_future": torch.tensor(power[context_end:end], dtype=torch.float32),
        }

        if self.use_covariates:
            temperature = df["temperature"].values.astype(np.float32)
            irradiance = df["irradiance"].values.astype(np.float32)

            result["temperature_context"] = torch.tensor(
                temperature[start:context_end], dtype=torch.float32
            )
            result["temperature_future"] = torch.tensor(
                temperature[context_end:end], dtype=torch.float32
            )
            result["irradiance_context"] = torch.tensor(
                irradiance[start:context_end], dtype=torch.float32
            )
            result["irradiance_future"] = torch.tensor(
                irradiance[context_end:end], dtype=torch.float32
            )

        return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader for training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
