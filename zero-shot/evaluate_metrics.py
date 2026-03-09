"""
Evaluation Metrics for Time Series Forecasting

Calculates comprehensive metrics for zero-shot forecasting results.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)"""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def mape_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error for non-zero y_true points (%)"""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def mean_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE of the means: |mean(y_true) - mean(y_pred)|"""
    return np.abs(np.mean(y_true) - np.mean(y_pred))


def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized MAE (MAE / mean of actual)"""
    mean_actual = np.mean(y_true)
    if mean_actual == 0:
        return np.nan
    return mae(y_true, y_pred) / mean_actual


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized RMSE (RMSE / mean of actual)"""
    mean_actual = np.mean(y_true)
    if mean_actual == 0:
        return np.nan
    return rmse(y_true, y_pred) / mean_actual


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray = None, seasonality: int = 1) -> float:
    """
    Mean Absolute Scaled Error

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_train: Training data for computing naive forecast error (optional)
        seasonality: Seasonality period for naive forecast (default: 1 for naive persistence)
    """
    if y_train is None:
        y_train = y_true

    # Naive forecast error (seasonal naive)
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    naive_mae = np.mean(naive_errors)

    if naive_mae == 0:
        return np.nan

    return mae(y_true, y_pred) / naive_mae


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (Coefficient of Determination)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """
    Quantile Loss (Pinball Loss)

    Args:
        y_true: Ground truth values
        y_pred: Predicted quantile values
        q: Quantile level (0-1)
    """
    errors = y_true - y_pred
    return np.mean(np.maximum(q * errors, (q - 1) * errors))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Prediction Interval Coverage (%)

    Args:
        y_true: Ground truth values
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
    """
    covered = (y_true >= lower) & (y_true <= upper)
    return np.mean(covered) * 100


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average width of prediction interval"""
    return np.mean(upper - lower)


def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.2) -> float:
    """
    Winkler Score for prediction intervals

    Lower is better. Penalizes both wide intervals and non-coverage.

    Args:
        y_true: Ground truth values
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
        alpha: Significance level (e.g., 0.2 for 80% interval)
    """
    width = upper - lower
    penalty_lower = (2 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return np.mean(width + penalty_lower + penalty_upper)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray = None,
    upper: np.ndarray = None,
    y_train: np.ndarray = None,
    seasonality: int = 48,  # 24 hours for 30-min data
) -> Dict[str, float]:
    """
    Calculate all forecasting metrics.

    Args:
        y_true: Ground truth values
        y_pred: Point predictions
        lower: Lower bound of prediction interval (optional)
        upper: Upper bound of prediction interval (optional)
        y_train: Training data for MASE calculation (optional)
        seasonality: Seasonality period for MASE

    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'MAE': mae(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'sMAPE': smape(y_true, y_pred),
        'MAPE_nonzero': mape_nonzero(y_true, y_pred),
        'Mean_MAE': mean_mae(y_true, y_pred),
        'NMAE': nmae(y_true, y_pred),
        'NRMSE': nrmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
    }

    # MASE if training data available
    if y_train is not None and len(y_train) > seasonality:
        metrics['MASE'] = mase(y_true, y_pred, y_train, seasonality)

    # Interval metrics if bounds provided
    if lower is not None and upper is not None:
        metrics['Coverage_80%'] = coverage(y_true, lower, upper)
        metrics['Interval_Width'] = interval_width(lower, upper)
        metrics['Winkler_Score'] = winkler_score(y_true, lower, upper, alpha=0.2)

    return metrics


def calculate_metrics_from_df(
    pred_df: pd.DataFrame,
    test_df: pd.DataFrame,
    context_df: pd.DataFrame = None,
    seasonality: int = 48,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Calculate metrics from prediction and test DataFrames.

    Args:
        pred_df: Predictions with columns [item_id, timestamp, predictions, 0.1, 0.9]
        test_df: Ground truth with columns [item_id, timestamp, power]
        context_df: Historical data for MASE calculation (optional)
        seasonality: Seasonality period

    Returns:
        Tuple of (overall_metrics, per_series_metrics_df)
    """
    # Merge predictions with ground truth
    merged = pred_df.merge(
        test_df[['item_id', 'timestamp', 'power']],
        on=['item_id', 'timestamp'],
        how='inner'
    )

    if len(merged) == 0:
        print("Warning: No matching timestamps found")
        return {}, pd.DataFrame()

    # Overall metrics
    y_true = merged['power'].values
    y_pred = merged['predictions'].values
    lower = merged['0.1'].values if '0.1' in merged.columns else None
    upper = merged['0.9'].values if '0.9' in merged.columns else None

    # Get training data if available
    y_train = None
    if context_df is not None:
        y_train = context_df['power'].values

    overall_metrics = calculate_all_metrics(y_true, y_pred, lower, upper, y_train, seasonality)
    overall_metrics['n_samples'] = len(merged)
    overall_metrics['n_series'] = merged['item_id'].nunique()

    # Pre-group context_df for faster lookup
    context_grouped = None
    if context_df is not None:
        context_grouped = {k: v['power'].values for k, v in context_df.groupby('item_id')}

    # Per-series metrics
    per_series_results = []
    for item_id, group in merged.groupby('item_id'):
        yt = group['power'].values
        yp = group['predictions'].values
        lo = group['0.1'].values if '0.1' in group.columns else None
        hi = group['0.9'].values if '0.9' in group.columns else None

        # Get training data for this series (fast lookup)
        yt_train = None
        if context_grouped is not None and item_id in context_grouped:
            yt_train = context_grouped[item_id]

        series_metrics = calculate_all_metrics(yt, yp, lo, hi, yt_train, seasonality)
        series_metrics['item_id'] = item_id
        series_metrics['n_samples'] = len(group)
        per_series_results.append(series_metrics)

    per_series_df = pd.DataFrame(per_series_results)
    # Reorder columns
    cols = ['item_id', 'n_samples'] + [c for c in per_series_df.columns if c not in ['item_id', 'n_samples']]
    per_series_df = per_series_df[cols]

    return overall_metrics, per_series_df


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print('=' * 60)
    for name, value in metrics.items():
        if isinstance(value, float):
            if np.isnan(value):
                print(f"{name:20s}: NaN")
            elif abs(value) < 0.01 or abs(value) > 1000:
                print(f"{name:20s}: {value:.4e}")
            else:
                print(f"{name:20s}: {value:.4f}")
        else:
            print(f"{name:20s}: {value}")


def main():
    """Evaluate metrics from saved predictions."""
    import os

    OUTPUT_DIR = "/home/ec2-user/efs/to_customer/zendure/zero-shot"
    DATA_DIR = "/home/ec2-user/efs/to_customer/zendure/data"

    # Load predictions
    pred_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    if not os.path.exists(pred_path):
        print(f"Predictions file not found: {pred_path}")
        print("Please run chronos_zero_shot.py first.")
        return

    pred_df = pd.read_csv(pred_path, parse_dates=['timestamp'])
    print(f"Loaded {len(pred_df)} predictions")

    # Load test data
    split_path = os.path.join(OUTPUT_DIR, "data_split.csv")
    split_df = pd.read_csv(split_path)
    test_files = split_df[split_df['split'] == 'test']['filename'].tolist()

    # Load test time series
    from pathlib import Path
    from tqdm import tqdm

    print(f"Loading {len(test_files)} test time series...")
    dfs = []
    for f in tqdm(test_files):
        file_path = os.path.join(DATA_DIR, f)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['date'])
            df = df.rename(columns={'date': 'timestamp'})
            df['item_id'] = Path(f).stem
            dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    # Split context and test
    max_timestamp = full_df['timestamp'].max()
    cutoff = max_timestamp - pd.Timedelta(days=7)
    context_df = full_df[full_df['timestamp'] <= cutoff]
    test_df = full_df[full_df['timestamp'] > cutoff]

    print(f"Context samples: {len(context_df)}, Test samples: {len(test_df)}")

    # Calculate metrics
    overall_metrics, per_series_df = calculate_metrics_from_df(
        pred_df, test_df, context_df, seasonality=48
    )

    # Print overall metrics
    print_metrics(overall_metrics, "Overall Metrics")

    # Save per-series metrics
    per_series_path = os.path.join(OUTPUT_DIR, "metrics_per_series.csv")
    per_series_df.to_csv(per_series_path, index=False)
    print(f"\nPer-series metrics saved to: {per_series_path}")

    # Print per-series summary
    print(f"\n{'=' * 60}")
    print("Per-Series Metrics Summary")
    print('=' * 60)
    summary_cols = ['MAE', 'RMSE', 'sMAPE', 'MAPE_nonzero', 'Mean_MAE', 'NMAE', 'NRMSE', 'R2']
    summary_cols = [c for c in summary_cols if c in per_series_df.columns]
    print(per_series_df[summary_cols].describe())

    # Save overall metrics
    overall_path = os.path.join(OUTPUT_DIR, "metrics_overall.csv")
    pd.DataFrame([overall_metrics]).to_csv(overall_path, index=False)
    print(f"\nOverall metrics saved to: {overall_path}")


if __name__ == "__main__":
    main()
