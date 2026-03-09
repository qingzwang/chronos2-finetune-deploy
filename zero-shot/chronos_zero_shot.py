"""
Chronos Zero-Shot Forecasting Script for Power Prediction

This script performs zero-shot time series forecasting using Chronos models.
Supports: Chronos-2 (with covariates), Chronos-Bolt, Chronos1 (without covariates).
Only runs on test set files defined in data_split.csv.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import evaluation metrics
from evaluate_metrics import (
    calculate_metrics_from_df,
    print_metrics,
)


def load_test_files(split_csv: str) -> list:
    """Load test file list from split CSV."""
    df = pd.read_csv(split_csv)
    test_files = df[df['split'] == 'test']['filename'].tolist()
    return test_files


def load_single_series(file_path: str) -> pd.DataFrame:
    """Load a single time series from CSV file."""
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.rename(columns={'date': 'timestamp'})
    df['item_id'] = Path(file_path).stem
    return df


def load_test_data(data_dir: str, test_files: list) -> pd.DataFrame:
    """
    Load only test time series data.

    Args:
        data_dir: Path to the directory containing CSV files
        test_files: List of test file names

    Returns:
        DataFrame with columns: item_id, timestamp, power, temperature, irradiance
    """
    data_path = Path(data_dir)

    print(f"Loading {len(test_files)} test time series...")

    dfs = []
    for f in tqdm(test_files):
        file_path = data_path / f
        if file_path.exists():
            dfs.append(load_single_series(file_path))

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(df)}")
    return df


def detect_missing_timesteps(df: pd.DataFrame, freq: str = '30min') -> dict:
    """
    Detect missing timesteps in each series.

    Args:
        df: DataFrame with item_id and timestamp columns
        freq: Expected frequency (e.g., '30min', '1H')

    Returns:
        Dictionary with missing info per series
    """
    missing_info = {}
    freq_delta = pd.Timedelta(freq)

    for item_id, group in df.groupby('item_id'):
        group = group.sort_values('timestamp')
        timestamps = group['timestamp'].values

        # Calculate actual gaps
        diffs = np.diff(timestamps)
        expected_diff = np.timedelta64(freq_delta)

        # Find gaps larger than expected
        gaps = diffs > expected_diff
        n_missing = 0

        if gaps.any():
            for i, is_gap in enumerate(gaps):
                if is_gap:
                    gap_size = (diffs[i] / expected_diff) - 1
                    n_missing += int(gap_size)

        if n_missing > 0:
            missing_info[item_id] = {
                'n_missing': n_missing,
                'original_length': len(group),
                'expected_length': len(group) + n_missing,
                'missing_pct': n_missing / (len(group) + n_missing) * 100
            }

    return missing_info


def resample_and_fill(df: pd.DataFrame, freq: str = '30min', fill_method: str = 'linear') -> pd.DataFrame:
    """
    Resample each series to regular frequency and fill missing values.

    Args:
        df: DataFrame with item_id, timestamp, and value columns
        freq: Target frequency (e.g., '30min', '1H')
        fill_method: 'linear' for interpolation, 'ffill' for forward fill

    Returns:
        DataFrame with regular timesteps and filled values
    """
    resampled_dfs = []

    for item_id, group in tqdm(df.groupby('item_id'), desc="Resampling series"):
        group = group.sort_values('timestamp').set_index('timestamp')

        # Get time range
        start = group.index.min()
        end = group.index.max()

        # Create regular time index
        regular_index = pd.date_range(start=start, end=end, freq=freq)

        # Reindex to regular frequency
        group_resampled = group.reindex(regular_index)

        # Fill missing values
        numeric_cols = group_resampled.select_dtypes(include=[np.number]).columns
        if fill_method == 'linear':
            group_resampled[numeric_cols] = group_resampled[numeric_cols].interpolate(method='linear')
        elif fill_method == 'ffill':
            group_resampled[numeric_cols] = group_resampled[numeric_cols].ffill()

        # Fill any remaining NaN at edges
        group_resampled[numeric_cols] = group_resampled[numeric_cols].bfill().ffill()

        # Reset index and add item_id
        group_resampled = group_resampled.reset_index().rename(columns={'index': 'timestamp'})
        group_resampled['item_id'] = item_id

        resampled_dfs.append(group_resampled)

    result = pd.concat(resampled_dfs, ignore_index=True)
    return result


def filter_short_series(
    df: pd.DataFrame,
    min_length: int,
) -> pd.DataFrame:
    """
    Filter out series that are shorter than the minimum required length.

    Args:
        df: DataFrame with item_id column
        min_length: Minimum number of data points required

    Returns:
        Filtered DataFrame
    """
    # Count length per series
    series_lengths = df.groupby('item_id').size()

    # Find series that meet minimum length requirement
    valid_series = series_lengths[series_lengths >= min_length].index.tolist()
    short_series = series_lengths[series_lengths < min_length].index.tolist()

    if len(short_series) > 0:
        print(f"Filtered out {len(short_series)} series with length < {min_length}:")
        for sid in short_series:
            print(f"  - {sid}: {series_lengths[sid]} points")

    # Filter dataframe
    filtered_df = df[df['item_id'].isin(valid_series)].copy()

    print(f"Remaining series: {len(valid_series)} / {len(series_lengths)}")
    print(f"Remaining records: {len(filtered_df)} / {len(df)}")

    return filtered_df


def split_context_forecast(df: pd.DataFrame, forecast_days: int = 7, cutoff: pd.Timestamp = None) -> tuple:
    """
    Split each time series into context (history) and forecast (future) periods.

    Args:
        df: Full dataframe
        forecast_days: Number of days to forecast
        cutoff: Cutoff timestamp (if None, use max_timestamp - forecast_days)

    Returns:
        Tuple of (context_df, test_df, future_df)
    """
    # Calculate cutoff timestamp (same for all series)
    if cutoff is None:
        max_timestamp = df['timestamp'].max()
        cutoff = max_timestamp - pd.Timedelta(days=forecast_days)

    forecast_end = cutoff + pd.Timedelta(days=forecast_days)

    context_df = df[df['timestamp'] <= cutoff].copy()
    test_df = df[(df['timestamp'] > cutoff) & (df['timestamp'] <= forecast_end)].copy()

    # Future df contains covariates for prediction period (no target)
    future_df = test_df[['item_id', 'timestamp', 'temperature', 'irradiance']].copy()

    return context_df, test_df, future_df


def generate_sliding_windows_for_series(
    series_df: pd.DataFrame,
    forecast_days: int = 7,
    stride_days: int = 7,
    n_windows: int = None,
    min_context_days: int = 30,
) -> list:
    """
    Generate sliding window cutoff timestamps for a single series.

    Args:
        series_df: DataFrame for a single series
        forecast_days: Number of days to forecast per window
        stride_days: Number of days to slide between windows
        n_windows: Number of windows per series (if None, generate as many as possible)
        min_context_days: Minimum days of context required

    Returns:
        List of cutoff timestamps (from newest to oldest)
    """
    max_timestamp = series_df['timestamp'].max()
    min_timestamp = series_df['timestamp'].min()

    # First cutoff: leave forecast_days at the end
    first_cutoff = max_timestamp - pd.Timedelta(days=forecast_days)

    # Minimum cutoff: need at least min_context_days of history
    min_cutoff = min_timestamp + pd.Timedelta(days=min_context_days)

    cutoffs = []
    current_cutoff = first_cutoff

    while current_cutoff >= min_cutoff:
        cutoffs.append(current_cutoff)
        if n_windows is not None and len(cutoffs) >= n_windows:
            break
        current_cutoff -= pd.Timedelta(days=stride_days)

    return cutoffs


def split_single_series(
    series_df: pd.DataFrame,
    cutoff: pd.Timestamp,
    forecast_days: int,
    context_length: int = None,
) -> tuple:
    """
    Split a single series into context, test, and future parts.

    Args:
        series_df: DataFrame for a single series
        cutoff: Cutoff timestamp
        forecast_days: Number of days to forecast
        context_length: Max context length (None = use all)

    Returns:
        Tuple of (context_df, test_df, future_df)
    """
    forecast_end = cutoff + pd.Timedelta(days=forecast_days)

    context_df = series_df[series_df['timestamp'] <= cutoff].copy()
    test_df = series_df[(series_df['timestamp'] > cutoff) & (series_df['timestamp'] <= forecast_end)].copy()

    # Truncate context if needed
    if context_length is not None and len(context_df) > context_length:
        context_df = context_df.tail(context_length).copy()

    # Future df contains covariates for prediction period
    if 'temperature' in test_df.columns and 'irradiance' in test_df.columns:
        future_df = test_df[['item_id', 'timestamp', 'temperature', 'irradiance']].copy()
    else:
        future_df = None

    return context_df, test_df, future_df


def get_model_type(model_name: str) -> str:
    """
    Determine model type from model name.

    Returns:
        'chronos2', 'chronos-bolt', or 'chronos1'
    """
    model_name_lower = model_name.lower()
    if 'chronos-2' in model_name_lower or 'chronos2' in model_name_lower:
        return 'chronos2'
    elif 'chronos-bolt' in model_name_lower or 'chronos_bolt' in model_name_lower:
        return 'chronos-bolt'
    else:
        # Default to chronos1 for chronos-t5-* models
        return 'chronos1'


def load_model(model_name: str = "amazon/chronos-2", device: str = "cuda"):
    """
    Load and return the Chronos pipeline.

    Supported models:
        - Chronos2: "amazon/chronos-2" (supports covariates)
        - Chronos-Bolt: "amazon/chronos-bolt-tiny", "amazon/chronos-bolt-mini",
                        "amazon/chronos-bolt-small", "amazon/chronos-bolt-base"
        - Chronos1: "amazon/chronos-t5-tiny", "amazon/chronos-t5-mini",
                    "amazon/chronos-t5-small", "amazon/chronos-t5-base",
                    "amazon/chronos-t5-large"
    """
    from chronos import BaseChronosPipeline

    model_type = get_model_type(model_name)
    print(f"Loading model: {model_name} (type: {model_type})")

    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=device
    )
    return pipeline, model_type


def run_zero_shot_forecast(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame = None,
    prediction_length: int = 48,
    context_length: int = None,
    pipeline=None,
    model_name: str = "amazon/chronos-2",
    model_type: str = None,
    use_covariates: bool = True,
    device: str = "cuda",
    batch_size: int = 100,
):
    """
    Run zero-shot forecasting using Chronos models.

    Args:
        context_df: Historical data
        future_df: Future covariates (optional, only for Chronos2)
        prediction_length: Number of steps to predict
        context_length: Number of historical steps to use (None = use all)
        pipeline: Pre-loaded model pipeline (if None, will load)
        model_name: Model name
        model_type: Model type ('chronos2', 'chronos-bolt', 'chronos1')
        use_covariates: Whether to use covariates (only for Chronos2)
        device: Device to run on
        batch_size: Number of series to process at once
    """
    if pipeline is None:
        pipeline, model_type = load_model(model_name, device)

    if model_type is None:
        model_type = get_model_type(model_name)

    # Chronos-Bolt and Chronos1 do not support covariates
    supports_covariates = (model_type == 'chronos2')
    if use_covariates and not supports_covariates:
        print(f"  Warning: {model_type} does not support covariates. Using univariate mode.")
        use_covariates = False

    if use_covariates and supports_covariates:
        input_df = context_df[['item_id', 'timestamp', 'power', 'temperature', 'irradiance']].copy()
    else:
        input_df = context_df[['item_id', 'timestamp', 'power']].copy()
        future_df = None

    # Truncate to context_length if specified
    if context_length is not None:
        input_df = input_df.groupby('item_id').tail(context_length).reset_index(drop=True)

    # Ensure future_df has the same series IDs as input_df (only for Chronos2 with covariates)
    if future_df is not None and supports_covariates:
        input_ids = set(input_df['item_id'].unique())
        future_ids = set(future_df['item_id'].unique())
        common_ids = input_ids & future_ids

        if len(common_ids) < len(input_ids):
            input_df = input_df[input_df['item_id'].isin(common_ids)].copy()
        if len(common_ids) < len(future_ids):
            future_df = future_df[future_df['item_id'].isin(common_ids)].copy()

        if len(common_ids) == 0:
            print("Warning: No common series between context and future data")
            return pd.DataFrame()

        # Filter out series with incomplete future data (must have prediction_length points)
        future_lengths = future_df.groupby('item_id').size()
        valid_future_ids = future_lengths[future_lengths >= prediction_length].index.tolist()

        if len(valid_future_ids) < len(common_ids):
            n_filtered = len(common_ids) - len(valid_future_ids)
            print(f"  Filtered {n_filtered} series with incomplete future data")
            input_df = input_df[input_df['item_id'].isin(valid_future_ids)].copy()
            future_df = future_df[future_df['item_id'].isin(valid_future_ids)].copy()
            # Keep only prediction_length points per series
            future_df = future_df.groupby('item_id').head(prediction_length).reset_index(drop=True)

        if len(valid_future_ids) == 0:
            print("Warning: No series with complete future data")
            return pd.DataFrame()

    # Get all unique series IDs
    all_ids = input_df['item_id'].unique()
    n_series = len(all_ids)

    # Build predict_df kwargs based on model type
    if model_type == 'chronos2' and future_df is not None:
        # Chronos2 with covariates
        predict_kwargs = {
            'future_df': future_df,
        }
    else:
        # Chronos-Bolt, Chronos1, or Chronos2 without covariates
        predict_kwargs = {}

    # Process in batches if needed
    if n_series <= batch_size:
        # Single batch
        pred_df = pipeline.predict_df(
            input_df,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column='item_id',
            timestamp_column='timestamp',
            target='power',
            **predict_kwargs,
        )
    else:
        # Multiple batches
        print(f"  Processing {n_series} series in batches of {batch_size}...")
        all_preds = []
        for i in range(0, n_series, batch_size):
            batch_ids = all_ids[i:i+batch_size]
            batch_input = input_df[input_df['item_id'].isin(batch_ids)]
            batch_kwargs = predict_kwargs.copy()
            if 'future_df' in batch_kwargs and batch_kwargs['future_df'] is not None:
                batch_kwargs['future_df'] = future_df[future_df['item_id'].isin(batch_ids)]

            print(f"    Batch {i//batch_size + 1}/{(n_series + batch_size - 1)//batch_size}: {len(batch_ids)} series")

            batch_pred = pipeline.predict_df(
                batch_input,
                prediction_length=prediction_length,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column='item_id',
                timestamp_column='timestamp',
                target='power',
                **batch_kwargs,
            )
            all_preds.append(batch_pred)

        print(f"  Concatenating {len(all_preds)} batch results...")
        pred_df = pd.concat(all_preds, ignore_index=True)
        print(f"  Done. Total predictions: {len(pred_df)}")

    return pred_df


def plot_forecast_example(
    context_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    item_id: str,
    history_length: int = 96,
    save_path: str = None,
):
    """Plot forecast for a single time series."""
    ctx = context_df[context_df['item_id'] == item_id].set_index('timestamp')['power'].tail(history_length)
    truth = test_df[test_df['item_id'] == item_id].set_index('timestamp')['power']
    pred = pred_df[pred_df['item_id'] == item_id].set_index('timestamp')

    fig, ax = plt.subplots(figsize=(14, 5))

    ctx.plot(ax=ax, label='Historical', color='blue', alpha=0.7)
    truth.plot(ax=ax, label='Ground Truth', color='green', alpha=0.7)

    if 'predictions' in pred.columns:
        pred['predictions'].plot(ax=ax, label='Forecast', color='red', linewidth=2)
        if '0.1' in pred.columns and '0.9' in pred.columns:
            ax.fill_between(pred.index, pred['0.1'], pred['0.9'],
                          alpha=0.3, color='red', label='80% Interval')

    ax.axvline(x=ctx.index[-1], color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Power')
    ax.set_title(f'Zero-Shot Forecast - {item_id}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Chronos Zero-Shot Forecasting for Power Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument('--data-dir', type=str,
                        default='/home/ubuntu/efs/to_customer/zendure/data',
                        help='Directory containing CSV data files')
    parser.add_argument('--output-dir', type=str,
                        default='/home/ubuntu/efs/to_customer/zendure/zero-shot',
                        help='Directory to save outputs')
    parser.add_argument('--split-csv', type=str, default=None,
                        help='Path to data split CSV (default: OUTPUT_DIR/data_split.csv)')

    # Model settings
    parser.add_argument('--model', type=str, default='amazon/chronos-2',
                        help='Model name. Options: amazon/chronos-2, '
                             'amazon/chronos-bolt-{tiny,mini,small,base}, '
                             'amazon/chronos-t5-{tiny,mini,small,base,large}')
    parser.add_argument('--use-covariates', action='store_true', default=True,
                        help='Use covariates (temperature, irradiance). Only effective for Chronos2')
    parser.add_argument('--no-covariates', action='store_true',
                        help='Disable covariates')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')

    # Forecast settings
    parser.add_argument('--context-days', type=float, default=1,
                        help='Number of days of historical context')
    parser.add_argument('--forecast-days', type=float, default=1,
                        help='Number of days to forecast per window')

    # Sliding window settings
    parser.add_argument('--stride-days', type=float, default=1,
                        help='Number of days between sliding windows')
    parser.add_argument('--n-windows', type=int, default=10,
                        help='Number of windows per series (0 = as many as possible)')
    parser.add_argument('--min-context-days', type=float, default=None,
                        help='Minimum context days required (default: same as context-days)')

    # Other settings
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of series to process at once')
    parser.add_argument('--seasonality', type=int, default=48,
                        help='Seasonality for metrics (48 = 24h for 30-min data)')
    parser.add_argument('--freq', type=str, default='30min',
                        help='Data frequency')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')

    return parser.parse_args()


def main():
    """Main function to run zero-shot forecasting with sliding windows."""

    # Parse arguments
    args = parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Configuration from args
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    SPLIT_CSV = args.split_csv if args.split_csv else os.path.join(OUTPUT_DIR, "data_split.csv")

    # Forecast settings
    CONTEXT_DAYS = args.context_days
    FORECAST_DAYS = args.forecast_days
    PREDICTION_LENGTH = int(48 * FORECAST_DAYS)  # 30-min intervals
    CONTEXT_LENGTH = int(48 * CONTEXT_DAYS)

    # Sliding window settings
    STRIDE_DAYS = args.stride_days
    N_WINDOWS = args.n_windows if args.n_windows > 0 else None
    MIN_CONTEXT_DAYS = args.min_context_days if args.min_context_days else CONTEXT_DAYS

    # Model settings
    MODEL_NAME = args.model
    USE_COVARIATES = args.use_covariates and not args.no_covariates
    SEASONALITY = args.seasonality
    BATCH_SIZE = args.batch_size
    FREQ = args.freq

    # Print configuration
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"  Model:           {MODEL_NAME}")
    print(f"  Use covariates:  {USE_COVARIATES}")
    print(f"  Device:          {args.device}")
    print(f"  Data dir:        {DATA_DIR}")
    print(f"  Output dir:      {OUTPUT_DIR}")
    print(f"  Context days:    {CONTEXT_DAYS} ({CONTEXT_LENGTH} steps)")
    print(f"  Forecast days:   {FORECAST_DAYS} ({PREDICTION_LENGTH} steps)")
    print(f"  Stride days:     {STRIDE_DAYS}")
    print(f"  N windows:       {N_WINDOWS}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Frequency:       {FREQ}")

    # Load test file list
    print("=" * 60)
    print("Loading Test File List")
    print("=" * 60)
    test_files = load_test_files(SPLIT_CSV)
    print(f"Test files: {len(test_files)}")

    # Load test data
    print("\n" + "=" * 60)
    print("Loading Test Data")
    print("=" * 60)
    df = load_test_data(DATA_DIR, test_files)

    # Detect missing timesteps
    print("\n" + "=" * 60)
    print("Detecting Missing Timesteps")
    print("=" * 60)
    missing_info = detect_missing_timesteps(df, freq=FREQ)

    if len(missing_info) > 0:
        print(f"Series with missing timesteps: {len(missing_info)}")
        total_missing = sum(info['n_missing'] for info in missing_info.values())
        print(f"Total missing timesteps: {total_missing}")

        # Show top 10 worst series
        sorted_missing = sorted(missing_info.items(), key=lambda x: x[1]['n_missing'], reverse=True)
        print(f"\nTop 10 series with most missing timesteps:")
        for item_id, info in sorted_missing[:10]:
            print(f"  {item_id}: {info['n_missing']} missing ({info['missing_pct']:.1f}%)")

        # Resample and fill
        print("\n" + "=" * 60)
        print("Resampling and Filling Missing Values")
        print("=" * 60)
        df = resample_and_fill(df, freq=FREQ, fill_method='linear')
        print(f"After resampling - Total records: {len(df)}")
    else:
        print("No missing timesteps detected.")

    # Filter out series that are too short
    print("\n" + "=" * 60)
    print("Filtering Short Series")
    print("=" * 60)
    min_required_length = CONTEXT_LENGTH + PREDICTION_LENGTH
    print(f"Minimum required length: {min_required_length} points")
    print(f"  (context: {CONTEXT_LENGTH} + forecast: {PREDICTION_LENGTH})")
    df = filter_short_series(df, min_length=min_required_length)

    # Print data statistics
    print("\n" + "=" * 60)
    print("Data Statistics")
    print("=" * 60)
    n_series = df['item_id'].nunique()
    series_lengths = df.groupby('item_id').size()
    print(df)
    print(f"Valid series: {n_series}")
    print(f"Total records: {len(df)}")
    print(f"Series length distribution:")
    print(f"  min:    {series_lengths.min()}")
    print(f"  max:    {series_lengths.max()}")
    print(f"  mean:   {series_lengths.mean():.1f}")
    print(f"  median: {series_lengths.median():.1f}")
    print(f"  std:    {series_lengths.std():.1f}")
    print(f"Length percentiles:")
    for p in [25, 50, 75, 90, 95]:
        print(f"  {p}%: {series_lengths.quantile(p/100):.0f}")

    if df['item_id'].nunique() == 0:
        print("ERROR: No series meet the minimum length requirement!")
        return

    # Group data by series
    print("\n" + "=" * 60)
    print("Preparing Per-Series Processing")
    print("=" * 60)
    series_groups = {item_id: group for item_id, group in df.groupby('item_id')}
    all_item_ids = list(series_groups.keys())
    print(f"Total series to process: {len(all_item_ids)}")

    # Generate cutoffs for each series and count total windows
    series_cutoffs = {}
    total_windows = 0
    for item_id, series_df in series_groups.items():
        cutoffs = generate_sliding_windows_for_series(
            series_df,
            forecast_days=FORECAST_DAYS,
            stride_days=STRIDE_DAYS,
            n_windows=N_WINDOWS,
            min_context_days=MIN_CONTEXT_DAYS,
        )
        if len(cutoffs) > 0:
            series_cutoffs[item_id] = cutoffs
            total_windows += len(cutoffs)

    print(f"Series with valid windows: {len(series_cutoffs)}")
    print(f"Total windows across all series: {total_windows}")

    # Show window distribution
    windows_per_series = [len(c) for c in series_cutoffs.values()]
    print(f"Windows per series: min={min(windows_per_series)}, max={max(windows_per_series)}, mean={np.mean(windows_per_series):.1f}")

    # Load model once
    print("\n" + "=" * 60)
    print("Loading Model")
    print("=" * 60)
    pipeline, model_type = load_model(MODEL_NAME, device=args.device)

    # Check if covariates are supported
    supports_covariates = (model_type == 'chronos2')
    use_covariates = USE_COVARIATES and supports_covariates
    if USE_COVARIATES and not supports_covariates:
        print(f"Warning: {model_type} does not support covariates. Using univariate mode.")

    # Run forecasting for each series
    all_predictions = []
    all_test_data = []
    all_context_data = []
    series_metrics = []

    processed_windows = 0
    for item_id in tqdm(series_cutoffs.keys(), desc="Processing series"):
        series_df = series_groups[item_id]
        cutoffs = series_cutoffs[item_id]

        series_preds = []
        series_tests = []

        for window_idx, cutoff in enumerate(cutoffs):
            # Split this series for this window
            context_df, test_df, future_df = split_single_series(
                series_df,
                cutoff=cutoff,
                forecast_days=FORECAST_DAYS,
                context_length=CONTEXT_LENGTH,
            )

            # Skip if not enough test data
            if len(test_df) < PREDICTION_LENGTH:
                continue

            # Prepare input for prediction
            if use_covariates:
                input_df = context_df[['item_id', 'timestamp', 'power', 'temperature', 'irradiance']].copy()
            else:
                input_df = context_df[['item_id', 'timestamp', 'power']].copy()
                future_df = None

            # Run prediction for single series
            try:
                # Build predict_df kwargs based on model type
                predict_kwargs = {
                    'prediction_length': PREDICTION_LENGTH,
                    'quantile_levels': [0.1, 0.5, 0.9],
                    'id_column': 'item_id',
                    'timestamp_column': 'timestamp',
                    'target': 'power',
                }
                # Only Chronos2 supports future_df (covariates)
                if model_type == 'chronos2' and future_df is not None:
                    predict_kwargs['future_df'] = future_df.head(PREDICTION_LENGTH)

                pred_df = pipeline.predict_df(input_df, **predict_kwargs)
            except Exception as e:
                continue

            if pred_df is None or len(pred_df) == 0:
                continue

            # Add window info
            pred_df['window_idx'] = window_idx
            pred_df['cutoff'] = cutoff
            test_df = test_df.head(PREDICTION_LENGTH).copy()
            test_df['window_idx'] = window_idx
            test_df['cutoff'] = cutoff

            series_preds.append(pred_df)
            series_tests.append(test_df)
            processed_windows += 1

        # Aggregate results for this series
        if len(series_preds) > 0:
            series_pred_df = pd.concat(series_preds, ignore_index=True)
            series_test_df = pd.concat(series_tests, ignore_index=True)

            all_predictions.append(series_pred_df)
            all_test_data.append(series_test_df)

            # Calculate metrics for this series (across all its windows)
            metrics, _ = calculate_metrics_from_df(
                series_pred_df, series_test_df, series_df, seasonality=SEASONALITY
            )
            metrics['item_id'] = item_id
            metrics['n_windows'] = len(series_preds)
            series_metrics.append(metrics)

    print(f"\nProcessed {processed_windows} total windows")

    # Combine all results
    print("\n" + "=" * 60)
    print("Aggregating Results")
    print("=" * 60)

    if len(all_predictions) == 0:
        print("ERROR: No predictions were generated!")
        return

    all_pred_df = pd.concat(all_predictions, ignore_index=True)
    all_test_df = pd.concat(all_test_data, ignore_index=True)

    print(f"Total predictions: {len(all_pred_df)}")
    print(f"Total test samples: {len(all_test_df)}")
    print(f"Series with predictions: {len(series_metrics)}")

    # Save all predictions
    pred_path = os.path.join(OUTPUT_DIR, "predictions_all_windows.csv")
    all_pred_df.to_csv(pred_path, index=False)
    print(f"All predictions saved to: {pred_path}")

    # Save per-series metrics
    series_metrics_df = pd.DataFrame(series_metrics)
    series_metrics_path = os.path.join(OUTPUT_DIR, "metrics_per_series.csv")
    series_metrics_df.to_csv(series_metrics_path, index=False)
    print(f"Per-series metrics saved to: {series_metrics_path}")

    # Calculate overall metrics (across all series and windows)
    print("\n" + "=" * 60)
    print("Overall Metrics (All Series)")
    print("=" * 60)
    overall_metrics, _ = calculate_metrics_from_df(
        all_pred_df, all_test_df, df, seasonality=SEASONALITY
    )
    overall_metrics['n_series'] = len(series_metrics)
    overall_metrics['n_total_windows'] = processed_windows
    print_metrics(overall_metrics, "Overall Metrics")

    # Save overall metrics
    overall_path = os.path.join(OUTPUT_DIR, "metrics_overall.csv")
    pd.DataFrame([overall_metrics]).to_csv(overall_path, index=False)
    print(f"\nOverall metrics saved to: {overall_path}")

    # Print per-series summary
    print("\n" + "=" * 60)
    print("Per-Series Metrics Summary")
    print("=" * 60)
    summary_cols = ['MAE', 'RMSE', 'sMAPE', 'MAPE_nonzero', 'Mean_MAE', 'NMAE', 'NRMSE', 'R2', 'n_windows']
    summary_cols = [c for c in summary_cols if c in series_metrics_df.columns]
    print(series_metrics_df[summary_cols].describe())

    # Plot example forecasts
    print("\n" + "=" * 60)
    print("Plotting Example Forecasts")
    print("=" * 60)

    # Pick 3 random series to plot
    sample_ids = list(series_cutoffs.keys())[:3]

    for sample_id in sample_ids:
        # Get data for this series
        series_pred = all_pred_df[all_pred_df['item_id'] == sample_id]
        series_test = all_test_df[all_test_df['item_id'] == sample_id]
        series_context = series_groups[sample_id]

        if len(series_pred) == 0 or len(series_test) == 0:
            continue

        # Plot the most recent window (window_idx == 0)
        last_pred = series_pred[series_pred['window_idx'] == 0]
        last_test = series_test[series_test['window_idx'] == 0]

        if len(last_pred) > 0 and len(last_test) > 0:
            plot_path = os.path.join(OUTPUT_DIR, f"forecast_{sample_id}.png")
            plot_forecast_example(series_context, last_test, last_pred, item_id=sample_id, save_path=plot_path)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Processed {len(series_metrics)} series, {processed_windows} total windows")
    print("=" * 60)


if __name__ == "__main__":
    main()
