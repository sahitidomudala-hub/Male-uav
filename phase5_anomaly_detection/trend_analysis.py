"""
Phase 5 extension: Reconstruction error trend analysis over mission timeline.

WHAT THIS DOES
    Characterizes HOW reconstruction error behaves over time, not just
    reports a single aggregate anomaly rate. Answers questions like: Are
    elevated errors clustered into a few discrete events or spread
    continuously? Is there a monotonic upward/downward drift? How does error
    evolve across the mission timeline?

WHAT THIS IS NOT (read this before presenting results)
    - This is NOT degradation modeling, NOT remaining-useful-life (RUL)
      estimation, and NOT a validated physics-based state estimator. It is
      DESCRIPTIVE trend reporting on reconstruction error only.
    - An "increasing trend" in reconstruction error does NOT prove physical
      degradation is occurring — it could be the aircraft transitioning
      into a different flight regime the model saw less of in training, or
      a sensor calibration drift, or a dozen other things that have nothing
      to do with actual engine health.
    - This analysis operates at the statistical-pattern level (what the
      LSTM autoencoder's reconstruction error does over time), not the
      physical-mechanism level (what the engine itself is doing). Mapping
      one to the other requires domain expertise, physics modeling, and
      operational context this script does not have.
    - RUL estimation and physics-based degradation tracking are the job of
      the Macro-Analytics layer downstream of this module, not this script.

METHOD
    1. Read outputs/reconstruction_errors_test.csv (one reconstruction_error
       value per sequence, in time order) and map each sequence index to its
       real timestamp using the same offset logic as create_downstream_handoff.py:
           test_start_row = 1 + train_rows + val_rows
           sequence i's start_row = test_start_row + i
       Each sequence's timestamp is taken as the start of its 50-sample window.

    2. Compute a rolling average of reconstruction error over a configurable
       window (default: 50 sequences, approximately 1 second of overlap-
       adjusted time at stride=1) to smooth out single-sequence noise.

    3. Fit a simple linear regression (error vs. time) and report slope +
       R² (or p-value) to quantify whether there's a statistically
       meaningful upward/downward drift. A flat or near-zero slope with low
       R² means "no trend"; a nonzero slope with high R² suggests monotonic
       drift (but NOT proven degradation — see caveats above).

    4. Classify whether elevated errors are concentrated in discrete
       clusters (reusing the consecutive-run-grouping logic from
       generate_alerts.py) or spread continuously. If clustered, report how
       many distinct clusters and their time ranges.

OUTPUT
    Single JSON file: outputs/trend_analysis.json with fields:
        {
          "rolling_window_size": <int>,
          "slope": <float>,  # reconstruction error per second
          "r_squared": <float>,
          "interpretation": "no significant trend" | "increasing trend" |
                            "decreasing trend" | "clustered, not continuous",
          "n_distinct_clusters": <int or null>,
          "cluster_time_ranges": [[start, end], ...] or null
        }

HOW TO READ THE RESULT
    - "no significant trend" + low R²: errors are statistically flat over
      time (anomalies may still exist, but they're not monotonically growing).
    - "increasing trend" + high R²: errors systematically rise over the
      mission timeline (could be degradation, could be flight regime shift,
      could be sensor drift — this script does not distinguish).
    - "clustered, not continuous": elevated errors happen in a few discrete
      time windows (like the known t~621s event), not as a gradual ongoing
      drift. This is the expected pattern for transient fault events.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline_config as _pc
import schema_guard as _sg

# Import the consecutive-sequence grouping logic from generate_alerts.py
# (don't reimplement it — reuse the exact same function for consistency).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_alerts import group_consecutive_sequences


def map_sequences_to_time(test_errors_df: pd.DataFrame, cleaned_df: pd.DataFrame,
                           test_start_row: int, stride: int = 1) -> pd.DataFrame:
    """Map each sequence index to its real start timestamp
    (cleaned row = test_start_row + sequence_index * stride)."""
    timestamps = []
    for seq_idx in test_errors_df.index:
        start_row = test_start_row + seq_idx * stride
        if start_row < len(cleaned_df):
            timestamps.append(cleaned_df.iloc[start_row]['Time'])
        else:
            timestamps.append(None)
    test_errors_df = test_errors_df.copy()
    test_errors_df['timestamp'] = timestamps
    return test_errors_df.dropna(subset=['timestamp'])


def compute_rolling_average(errors: np.ndarray, window: int) -> np.ndarray:
    """Simple centered rolling average (or trailing if you want causal-only)."""
    return pd.Series(errors).rolling(window=window, center=False, min_periods=1).mean().values


def fit_trend(timestamps: np.ndarray, errors: np.ndarray) -> dict:
    """Linear regression: error ~ time using numpy.polyfit. Returns slope and R²."""
    # Fit line: y = slope * x + intercept
    coeffs = np.polyfit(timestamps, errors, 1)
    slope, intercept = coeffs[0], coeffs[1]
    
    # Calculate R² manually
    y_pred = slope * timestamps + intercept
    ss_res = np.sum((errors - y_pred) ** 2)
    ss_tot = np.sum((errors - np.mean(errors)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_squared),
    }


def classify_pattern(errors: np.ndarray, threshold: float, slope: float,
                      r_squared: float, timestamps: np.ndarray,
                      r_squared_threshold: float = None,
                      slope_threshold: float = None) -> dict:
    """Decide: no trend, increasing/decreasing trend, or clustered."""
    r_squared_threshold = _pc.TREND_R_SQUARED_THRESHOLD if r_squared_threshold is None else r_squared_threshold
    slope_threshold = _pc.TREND_SLOPE_THRESHOLD if slope_threshold is None else slope_threshold
    # First, check if there's a significant monotonic trend
    if r_squared > r_squared_threshold and abs(slope) > slope_threshold:
        if slope > 0:
            interpretation = "increasing trend"
        else:
            interpretation = "decreasing trend"
        return {
            'interpretation': interpretation,
            'n_distinct_clusters': None,
            'cluster_time_ranges': None,
        }

    # If no strong trend, check if elevated errors are clustered
    anomalous_indices = np.where(errors > threshold)[0].tolist()
    if not anomalous_indices:
        return {
            'interpretation': "no significant trend",
            'n_distinct_clusters': 0,
            'cluster_time_ranges': [],
        }

    groups = group_consecutive_sequences(anomalous_indices)
    if len(groups) == 0:
        return {
            'interpretation': "no significant trend",
            'n_distinct_clusters': 0,
            'cluster_time_ranges': [],
        }

    # If anomalies are concentrated in a few discrete clusters (< 20% of
    # the total timeline filled with anomalies), call it "clustered, not continuous"
    total_anomalous = len(anomalous_indices)
    total_sequences = len(errors)
    if total_anomalous / total_sequences < 0.2 and len(groups) > 1:
        cluster_ranges = []
        for group in groups:
            start_time = timestamps[group[0]]
            end_time = timestamps[group[-1]]
            cluster_ranges.append([float(start_time), float(end_time)])
        return {
            'interpretation': "clustered, not continuous",
            'n_distinct_clusters': len(groups),
            'cluster_time_ranges': cluster_ranges,
        }

    # Otherwise, no clear pattern
    return {
        'interpretation': "no significant trend",
        'n_distinct_clusters': len(groups),
        'cluster_time_ranges': None,
    }


def main(base_dir: str = ".", rolling_window: int = None):
    rolling_window = _pc.TREND_ROLLING_WINDOW if rolling_window is None else rolling_window
    base = Path(base_dir)
    test_errors_path = base / "outputs" / "reconstruction_errors_test.csv"
    cleaned_path = base / "data" / "processed" / "cleaned_telemetry.csv"
    metadata_path = base / "data" / "sequences" / "metadata.json"
    report_path = base / "outputs" / "anomaly_detection_report.json"

    for p in [test_errors_path, cleaned_path, metadata_path, report_path]:
        if not p.exists():
            sys.exit(f"Missing required file: {p}")

    print("Loading reconstruction errors and metadata...")
    test_errors_df = pd.read_csv(test_errors_path)
    cleaned_df = pd.read_csv(cleaned_path)
    meta = _sg.verify_schema(base)
    report = json.load(open(report_path, encoding="utf-8-sig"))

    # first cleaned row of the test split = warm-up rows + train rows + val rows
    test_start_row = int(meta["warmup_rows_removed"]) + meta["train_rows"] + meta["val_rows"]
    threshold = report["threshold_calculation"]["threshold_value"]

    print(f"Mapping {len(test_errors_df)} test sequences to timestamps...")
    test_errors_df = map_sequences_to_time(test_errors_df, cleaned_df, test_start_row, int(meta["stride"]))

    errors = test_errors_df["reconstruction_error_mse"].values
    timestamps = test_errors_df["timestamp"].values

    print(f"Computing rolling average (window={rolling_window})...")
    rolling_errors = compute_rolling_average(errors, rolling_window)

    print("Fitting linear trend (error vs. time)...")
    trend = fit_trend(timestamps, errors)
    print(f"  Slope: {trend['slope']:.8f} (error per second)")
    print(f"  R²: {trend['r_squared']:.4f}")

    print("Classifying error pattern (clustered vs. continuous)...")
    pattern = classify_pattern(errors, threshold, trend['slope'], trend['r_squared'], timestamps)
    print(f"  Interpretation: {pattern['interpretation']}")
    if pattern['n_distinct_clusters'] is not None:
        print(f"  Distinct clusters: {pattern['n_distinct_clusters']}")

    result = {
        "rolling_window_size": rolling_window,
        "slope": trend["slope"],
        "r_squared": trend["r_squared"],
        **pattern,
    }

    out_path = base / "outputs" / "trend_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Trend analysis saved: {out_path}")

    print("\nREMINDER: This is descriptive trend reporting on reconstruction error only.")
    print("An 'increasing trend' does NOT prove physical degradation — it could be flight")
    print("regime shifts, sensor drift, or other non-degradation factors. RUL estimation")
    print("and physics-based degradation tracking are the job of the Macro-Analytics layer,")
    print("not this script.")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction error trend analysis over mission timeline")
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument("--rolling-window", type=int, default=_pc.TREND_ROLLING_WINDOW,
                        help=f"Rolling average window in sequences (default from pipeline_config: {_pc.TREND_ROLLING_WINDOW})")
    args = parser.parse_args()
    main(args.base_dir, args.rolling_window)
