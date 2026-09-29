"""
Phase 5 extension: Per-channel anomaly attribution.

WHAT THIS DOES
    For each anomalous sequence, ranks the physical telemetry channels by
    how much each one's reconstruction error deviates from ITS OWN normal
    (validation-derived) range. This upgrades "sequence N is anomalous" to
    "sequence N is anomalous, and FuelFlow/RPM look most responsible."

WHAT THIS IS NOT (read this before presenting results)
    - This is NOT sensor-fault isolation and NOT a validated root-cause
      diagnosis. It is a statistical deviation ranking: which signal's
      pattern moved furthest from what the model learned as normal.
    - It does NOT prove which sensor physically failed, or that the
      implicated channel caused the anomaly rather than responded to it.
    - No retraining, no architecture change, no new data -- this only
      reuses the (sequence_length, n_features) reconstruction the model already computes and
      stops collapsing it into a single number.
    - Rolled up from the model features to physical channels (e.g.
      EngTrq_raw / EngTrq_ma / EngTrq_std / EngTrq_rate all count as
      "EngTrq") because several of the model features are derived from the
      same underlying signal -- ranking at the raw feature level would make
      one real deviation look like "4 separate anomalies."

METHOD
    1. Per validation sequence, compute per-FEATURE squared error (average
       over the sequence timesteps only, NOT over features) -> (n_val, n_features) matrix.
    2. Roll up to per-CHANNEL error by averaging the features that belong to
       each physical channel.
    3. Per-channel threshold = 95th percentile of that channel's error
       across validation sequences (same principle as the existing overall
       threshold, just computed once per channel instead of once overall).
    4. For a given anomalous test sequence, compute the same per-channel
       errors and rank channels by (error / that channel's threshold) --
       a normalized "how many times over its own normal range" score, so
       channels with naturally larger/smaller error scales are comparable.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from channel_config import get_physical_channels

import pipeline_config as _pc
import schema_guard as _sg

_SUFFIXES = tuple(_pc.get_feature_suffixes().values())


def build_feature_to_channel(feature_names) -> dict:
    """Map every model feature back to its physical channel by stripping the
    single suffix definition in pipeline_config. Raises if a feature cannot
    be mapped to a configured channel (never guesses)."""
    channels = set(get_physical_channels())
    mapping = {}
    for f in feature_names:
        for sfx in _SUFFIXES:
            if f.endswith(sfx) and f[:-len(sfx)] in channels:
                mapping[f] = f[:-len(sfx)]
                break
        else:
            raise ValueError(f"Feature '{f}' does not map to any configured channel.")
    return mapping


def per_feature_errors(sequences: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """(n_seq, L, F) x2 -> (n_seq, F) squared error, averaged over TIME only."""
    return np.mean((sequences - reconstructions) ** 2, axis=1)


def rollup_to_channels(feature_errors: np.ndarray, feature_names: list) -> pd.DataFrame:
    """(n_seq, F) feature errors -> (n_seq, n_channels) channel errors, by
    averaging the features that belong to each physical channel."""
    df = pd.DataFrame(feature_errors, columns=feature_names)
    feature_to_channel = build_feature_to_channel(feature_names)
    channel_cols = {}
    for channel in get_physical_channels():
        member_features = [f for f, ch in feature_to_channel.items() if ch == channel]
        if member_features:
            channel_cols[channel] = df[member_features].mean(axis=1)
    return pd.DataFrame(channel_cols)


def compute_channel_thresholds(val_channel_errors: pd.DataFrame, percentile: float = None,
                                min_threshold_floor: float = None) -> pd.Series:
    """Per-channel threshold = percentile of THAT channel's validation error.
    Mirrors the existing overall-threshold methodology, just per channel.

    min_threshold_floor guards against near-constant channels (e.g.
    BaroPress, which is essentially flat in this dataset) whose validation
    error is so close to zero that dividing by it later would make the
    normalized ratio wildly unstable -- a tiny absolute error on a
    near-constant channel could otherwise register as "1000x over
    threshold" and drown out a genuinely large deviation elsewhere. The
    floor does not change which channels get flagged in absolute terms; it
    only keeps the RATIO used for ranking numerically sane for channels
    with almost no natural variability to measure a percentile from.
    """
    percentile = _pc.THRESHOLD_PERCENTILE if percentile is None else percentile
    min_threshold_floor = _pc.ATTRIBUTION_MIN_THRESHOLD_FLOOR if min_threshold_floor is None else min_threshold_floor
    raw_thresholds = val_channel_errors.quantile(percentile / 100.0)
    return raw_thresholds.clip(lower=min_threshold_floor)


def attribute_sequence(channel_errors_row: pd.Series, channel_thresholds: pd.Series, top_n: int = None) -> pd.DataFrame:
    """Rank channels for ONE sequence by how many times over their own
    threshold they are. Returns top_n channels, most-implicated first."""
    top_n = _pc.ATTRIBUTION_TOP_N if top_n is None else top_n
    ratio = channel_errors_row / channel_thresholds
    ranked = ratio.sort_values(ascending=False)
    return pd.DataFrame({
        'channel': ranked.index[:top_n],
        'error': channel_errors_row[ranked.index[:top_n]].values,
        'threshold': channel_thresholds[ranked.index[:top_n]].values,
        'times_over_threshold': ranked.values[:top_n],
    })


def run_attribution(model, X_val: np.ndarray, X_test: np.ndarray, feature_names: list,
                     test_error_path: str, output_path: str, percentile: float = None):
    """Full pipeline: compute channel thresholds from val, attribute every
    anomalous test sequence, save a CSV alongside the existing results."""
    print("Computing per-channel validation errors (for per-channel thresholds)...")
    val_recon = model.predict(X_val, batch_size=_pc.INFERENCE_BATCH_SIZE, verbose=0)
    val_feat_err = per_feature_errors(X_val, val_recon)
    val_channel_err = rollup_to_channels(val_feat_err, feature_names)
    channel_thresholds = compute_channel_thresholds(val_channel_err, percentile)
    print(f"Per-channel thresholds ({_pc.THRESHOLD_PERCENTILE if percentile is None else percentile}th percentile of validation error):")
    for ch, th in channel_thresholds.items():
        print(f"  {ch}: {th:.6f}")

    print("\nComputing per-channel test errors and attributing anomalies...")
    test_recon = model.predict(X_test, batch_size=_pc.INFERENCE_BATCH_SIZE, verbose=0)
    test_feat_err = per_feature_errors(X_test, test_recon)
    test_channel_err = rollup_to_channels(test_feat_err, feature_names)

    top_n = _pc.ATTRIBUTION_TOP_N
    rows = []
    for seq_idx in range(len(test_channel_err)):
        row = test_channel_err.iloc[seq_idx]
        ranked = attribute_sequence(row, channel_thresholds, top_n=top_n)
        rec = {'sequence_index': seq_idx}
        for k in range(min(top_n, len(ranked))):
            rec[f'top{k + 1}_channel'] = ranked.iloc[k]['channel']
            rec[f'top{k + 1}_times_over_threshold'] = ranked.iloc[k]['times_over_threshold']
        rows.append(rec)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    print(f"\n✓ Attribution saved: {output_path} ({len(result_df)} sequences)")
    return result_df, channel_thresholds


if __name__ == '__main__':
    import argparse
    import os
    os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
    import keras

    parser = argparse.ArgumentParser(description='Per-channel anomaly attribution (no retraining)')
    parser.add_argument('--base-dir', type=str, default='.')
    parser.add_argument('--percentile', type=float, default=_pc.THRESHOLD_PERCENTILE)
    args = parser.parse_args()

    base = Path(args.base_dir)
    meta = _sg.verify_schema(base)
    model = keras.models.load_model(base / 'models' / 'lstm_autoencoder.keras')
    X_val = np.load(base / 'data' / 'sequences' / 'X_val.npy')
    X_test = np.load(base / 'data' / 'sequences' / 'X_test.npy')

    run_attribution(
        model, X_val, X_test, meta['feature_names'],
        test_error_path=str(base / 'outputs' / 'reconstruction_errors_test.csv'),
        output_path=str(base / 'outputs' / 'anomaly_attribution.csv'),
        percentile=args.percentile,
    )
