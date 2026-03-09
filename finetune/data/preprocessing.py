"""
Data preprocessing utilities for Chronos fine-tuning.
"""

import os
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
import torch
from pathlib import Path


def csv_to_chronos2_input(
    csv_path: str,
    use_covariates: bool = True,
) -> Dict:
    """
    Convert a CSV file to Chronos-2 input format.

    Chronos-2 supports covariates in the format:
    {
        "target": tensor,
        "past_covariates": {"temperature": tensor, "irradiance": tensor},
        "future_covariates": {"temperature": tensor, "irradiance": tensor}
    }

    Args:
        csv_path: Path to CSV file
        use_covariates: Whether to include covariates

    Returns:
        Dictionary in Chronos-2 format
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    # Target (power)
    target = torch.tensor(df["power"].values, dtype=torch.float32)

    if not use_covariates:
        return {"target": target}

    # Covariates
    temperature = torch.tensor(df["temperature"].values, dtype=torch.float32)
    irradiance = torch.tensor(df["irradiance"].values, dtype=torch.float32)

    return {
        "target": target,
        "past_covariates": {
            "temperature": temperature,
            "irradiance": irradiance,
        },
        "future_covariates": {
            "temperature": temperature,
            "irradiance": irradiance,
        },
    }


def csv_to_univariate_input(csv_path: str) -> np.ndarray:
    """
    Convert a CSV file to univariate input (power only).

    Used for Chronos-1 and Chronos-Bolt which don't support covariates.

    Args:
        csv_path: Path to CSV file

    Returns:
        NumPy array of power values
    """
    df = pd.read_csv(csv_path)
    return df["power"].values.astype(np.float32)


def prepare_chronos1_dataset(
    file_paths: List[str],
    output_dir: str,
    context_length: int = 512,
    prediction_length: int = 48,
) -> str:
    """
    Prepare data in GluonTS format for Chronos-1 training.

    Chronos-1 training script expects data in GluonTS JSON format.

    Args:
        file_paths: List of CSV file paths
        output_dir: Directory to save processed data
        context_length: Context length for training
        prediction_length: Prediction length for training

    Returns:
        Path to the output directory containing train.json
    """
    import json
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train.json")

    min_length = context_length + prediction_length

    with open(output_path, "w") as f:
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                df["date"] = pd.to_datetime(df["date"])

                if len(df) < min_length:
                    continue

                # GluonTS format
                record = {
                    "start": df["date"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
                    "target": df["power"].values.tolist(),
                    "item_id": Path(file_path).stem,
                }

                f.write(json.dumps(record) + "\n")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    print(f"Saved GluonTS dataset to {output_path}")
    return output_dir


def normalize_series(
    series: np.ndarray,
    method: str = "standard"
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize a time series.

    Args:
        series: Input time series
        method: Normalization method ("standard", "minmax", "robust")

    Returns:
        Tuple of (normalized series, normalization parameters)
    """
    if method == "standard":
        loc = np.nanmean(series)
        scale = np.nanstd(series)
        scale = scale if scale > 1e-8 else 1.0
        normalized = (series - loc) / scale
        params = {"loc": loc, "scale": scale, "method": method}

    elif method == "minmax":
        min_val = np.nanmin(series)
        max_val = np.nanmax(series)
        range_val = max_val - min_val
        range_val = range_val if range_val > 1e-8 else 1.0
        normalized = (series - min_val) / range_val
        params = {"min": min_val, "max": max_val, "method": method}

    elif method == "robust":
        median = np.nanmedian(series)
        q1 = np.nanpercentile(series, 25)
        q3 = np.nanpercentile(series, 75)
        iqr = q3 - q1
        iqr = iqr if iqr > 1e-8 else 1.0
        normalized = (series - median) / iqr
        params = {"median": median, "iqr": iqr, "method": method}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params


def denormalize_series(
    series: np.ndarray,
    params: Dict
) -> np.ndarray:
    """
    Denormalize a time series.

    Args:
        series: Normalized time series
        params: Normalization parameters from normalize_series

    Returns:
        Denormalized series
    """
    method = params["method"]

    if method == "standard":
        return series * params["scale"] + params["loc"]
    elif method == "minmax":
        range_val = params["max"] - params["min"]
        return series * range_val + params["min"]
    elif method == "robust":
        return series * params["iqr"] + params["median"]
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_batches_for_chronos2(
    inputs: List[Dict],
    batch_size: int = 256,
) -> List[List[Dict]]:
    """
    Create batches of inputs for Chronos-2 training.

    Args:
        inputs: List of Chronos-2 format dictionaries
        batch_size: Number of samples per batch

    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        batches.append(batch)
    return batches


def collate_chronos_batch(
    batch: List[Dict],
    context_length: int,
    prediction_length: int,
    use_covariates: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of samples for Chronos training.

    Args:
        batch: List of sample dictionaries
        context_length: Context length
        prediction_length: Prediction length
        use_covariates: Whether to include covariates

    Returns:
        Batched tensors
    """
    target_context = torch.stack([s["target_context"] for s in batch])
    target_future = torch.stack([s["target_future"] for s in batch])

    result = {
        "target_context": target_context,
        "target_future": target_future,
    }

    if use_covariates:
        result["temperature_context"] = torch.stack(
            [s["temperature_context"] for s in batch]
        )
        result["temperature_future"] = torch.stack(
            [s["temperature_future"] for s in batch]
        )
        result["irradiance_context"] = torch.stack(
            [s["irradiance_context"] for s in batch]
        )
        result["irradiance_future"] = torch.stack(
            [s["irradiance_future"] for s in batch]
        )

    return result


def prepare_chronos2_inputs(
    file_paths: List[str],
    use_covariates: bool = True,
) -> List[Union[torch.Tensor, Dict]]:
    """
    Prepare inputs for Chronos-2 Pipeline.fit().

    Args:
        file_paths: List of CSV file paths
        use_covariates: Whether to include covariates

    Returns:
        List of inputs in Chronos-2 format
    """
    inputs = []

    for file_path in file_paths:
        try:
            input_data = csv_to_chronos2_input(file_path, use_covariates)
            inputs.append(input_data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    print(f"Prepared {len(inputs)} inputs for Chronos-2")
    return inputs


def prepare_bolt_tensors(
    file_paths: List[str],
    context_length: int = 512,
    prediction_length: int = 48,
    stride: int = 48,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare tensors for Chronos-Bolt training.

    Args:
        file_paths: List of CSV file paths
        context_length: Context length
        prediction_length: Prediction length
        stride: Stride for sliding window

    Returns:
        Tuple of (context tensors, target tensors)
    """
    contexts = []
    targets = []
    total_length = context_length + prediction_length

    for file_path in file_paths:
        try:
            power = csv_to_univariate_input(file_path)

            if len(power) < total_length:
                continue

            for start in range(0, len(power) - total_length + 1, stride):
                context_end = start + context_length
                end = start + total_length

                contexts.append(power[start:context_end])
                targets.append(power[context_end:end])

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    context_tensor = torch.tensor(np.array(contexts), dtype=torch.float32)
    target_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

    print(f"Prepared {len(contexts)} samples for Chronos-Bolt")
    return context_tensor, target_tensor
