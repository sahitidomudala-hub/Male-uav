"""
schema_guard.py
================
Fails LOUDLY when the current configuration no longer matches the trained
artifacts, instead of letting a stale model silently score data it was never
trained for.

WHAT IS CHECKED (verify_schema)
    1. A fingerprint of everything that changes the model's input or the way
       features are made (channel_config + the relevant pipeline_config
       values) equals the fingerprint saved by Phase 3 in metadata.json.
    2. The feature set implied by channel_config equals the feature names the
       model was trained on.
    3. models/filter_params.json exists and covers every configured channel.

The fingerprint deliberately excludes values that do not change the model
(severity tiers, uplink policy, bootstrap settings, fault-injection
scenarios...), so editing those never forces a retrain.

Everything that depends on the trained model -- feature names, sequence
length, sampling rate, warm-up rows, split boundaries, redundant channel
groups -- is READ from the saved metadata (returned by verify_schema), never
from config or literals.
"""

from __future__ import annotations  # Postponed annotations for TYPE_CHECKING

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd  # Only for type hints, not runtime

import channel_config as _cc
import pipeline_config as _pc

# pipeline_config values that change the trained model or its input features.
FINGERPRINT_PARAMS = [
    'SEQUENCE_LENGTH', 'STRIDE', 'TRAIN_RATIO', 'VAL_RATIO', 'TEST_RATIO',
    'MOVING_WINDOW_SECONDS', 'ROLLING_MEAN_MIN_PERIODS', 'ROLLING_STD_MIN_PERIODS',
    'FEATURE_SUFFIX_RAW', 'FEATURE_SUFFIX_MA', 'FEATURE_SUFFIX_STD', 'FEATURE_SUFFIX_RATE',
    'AUTO_CUTOFF_POWER_FRACTION', 'AUTO_CUTOFF_MIN_HZ', 'AUTO_CUTOFF_MAX_HZ',
    'AUTO_CUTOFF_FALLBACK_HZ', 'BUTTERWORTH_ORDER', 'NYQUIST_GUARD',
    'ROLLING_MEDIAN_MIN_PERIODS', 'CORRECTION_TOLERANCE_FRACTION_OF_RANGE',
    'FILTER_STEADY_STATE_INIT',
    'LSTM_UNITS', 'LEARNING_RATE', 'OPTIMIZER', 'LOSS',
]


class SchemaMismatchError(RuntimeError):
    """Raised when config and trained artifacts disagree (retrain required)."""


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return obj


def config_snapshot() -> dict:
    return {
        'channels': _jsonable(_cc.CHANNEL_CONFIG),
        'params': {name: _jsonable(getattr(_pc, name)) for name in FINGERPRINT_PARAMS},
    }


def config_fingerprint() -> str:
    blob = json.dumps(config_snapshot(), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def expected_feature_set() -> set:
    """Feature names implied by channel_config (order-insensitive)."""
    sfx = _pc.get_feature_suffixes()
    names = set()
    for ch, cfg in _cc.CHANNEL_CONFIG.items():
        if not cfg.get('model_input', True):
            continue
        names.add(ch + sfx['raw'])
        if cfg.get('moving_stats'):
            names.update({ch + sfx['ma'], ch + sfx['std']})
        if cfg.get('rate'):
            names.add(ch + sfx['rate'])
    return names


def compute_redundant_groups(train_df: pd.DataFrame, channels, cutoff: float = None) -> list:
    """
    Groups of model-input channels that carry (almost) the same information:
    |Pearson r| of their values on the TRAINING split >= cutoff.
    Returns a list of sorted channel-name lists (groups of size >= 2).
    Computed from training data only, so it cannot leak validation/test.
    """
    cutoff = _pc.REDUNDANCY_CORRELATION_CUTOFF if cutoff is None else cutoff
    cols = [c for c in channels if c in train_df.columns]
    corr = train_df[cols].corr().abs()
    parent = {c: c for c in cols}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if np.isfinite(r) and r >= cutoff:
                parent[find(a)] = find(b)

    groups = {}
    for c in cols:
        groups.setdefault(find(c), []).append(c)
    return [sorted(g) for g in groups.values() if len(g) >= 2]


def load_metadata(base_dir='.') -> dict:
    path = Path(base_dir) / 'data' / 'sequences' / 'metadata.json'
    if not path.exists():
        raise SchemaMismatchError(f"Training metadata not found: {path}. Run the pipeline through "
                                  "Phase 3 and train the model first.")
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def verify_schema(base_dir='.') -> dict:
    """Raise SchemaMismatchError if config and artifacts disagree; otherwise
    return the saved training metadata (the source of truth for everything
    tied to the trained model)."""
    base = Path(base_dir)
    meta = load_metadata(base)
    problems = []

    saved_fp = meta.get('config_fingerprint')
    if saved_fp is None:
        problems.append("metadata.json has no config_fingerprint (produced by an older pipeline version).")
    elif saved_fp != config_fingerprint():
        problems.append("channel_config / pipeline_config changed since the model was trained "
                        "(config fingerprint differs).")

    saved_features = set(meta.get('feature_names', []))
    expected = expected_feature_set()
    if saved_features != expected:
        only_saved = sorted(saved_features - expected)
        only_config = sorted(expected - saved_features)
        problems.append(f"Feature set differs. Trained but not in config: {only_saved}. "
                        f"In config but not trained: {only_config}.")

    fp_path = base / 'models' / _pc.FILTER_PARAMS_FILENAME
    if not fp_path.exists():
        problems.append(f"Saved filter parameters missing: {fp_path}.")
    else:
        with open(fp_path, 'r', encoding='utf-8-sig') as f:
            saved_filters = json.load(f)
        missing = [ch for ch in _cc.CHANNEL_CONFIG if ch not in saved_filters.get('channels', {})]
        if missing:
            problems.append(f"Saved filter parameters lack channels: {missing}.")

    if problems:
        raise SchemaMismatchError(
            "Configuration does not match the trained artifacts -- RETRAIN REQUIRED "
            "(run Phases 1-4 again) or restore the configuration used for training:\n  - "
            + "\n  - ".join(problems))
    return meta


def check_sampling_rate(fs_hz: float, meta: dict) -> None:
    """Reject inputs whose sampling rate differs from the training rate: all
    windows/filters are defined in samples/Hz of the training data."""
    train_fs = meta.get('sampling_rate_hz')
    if train_fs is None:
        raise SchemaMismatchError("metadata.json has no sampling_rate_hz; retrain required.")
    if abs(fs_hz - train_fs) > _pc.SAMPLING_RATE_TOLERANCE_FRACTION * train_fs:
        raise SchemaMismatchError(f"Input sampling rate {fs_hz:.3f} Hz differs from the training rate "
                                  f"{train_fs:.3f} Hz beyond tolerance.")
