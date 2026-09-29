"""
block_scoring.py
=================
Push a block of RAW telemetry through the REAL pipeline with the SAVED
artifacts and get reconstruction errors back. Used by the fault-injection and
noise-sweep validations so that injected faults / noise pass through exactly
what real data would: Phase 1 (loaded filter cutoffs, corrections), Phase 2
features, the saved scaler and the saved model. Nothing is refit.

Also holds the perturbation ops shared by both validations.
"""

import contextlib
import io
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for _sub in ('phase1_preprocessing', 'phase2_feature_extraction', 'phase5_anomaly_detection'):
    sys.path.insert(0, str(ROOT / _sub))
sys.path.insert(0, str(ROOT))

import channel_config as cc
import pipeline_config as pc
import schema_guard as sg
from preprocess import TelemetryPreprocessor
from extract_features import TelemetryFeatureExtractor
from attribute_anomalies import per_feature_errors, rollup_to_channels, compute_channel_thresholds
from inference_core import load_threshold


# --------------------------------------------------------------------------
# Perturbation ops. Each takes the values of the target window `x` (1-D),
# `pre` = the last real value before the window, and the scenario `value`.
# --------------------------------------------------------------------------
def apply_op(op: str, x: np.ndarray, pre: float, value, rng: np.random.Generator = None) -> np.ndarray:
    n = len(x)
    if op == 'step_offset':
        return x + value
    if op == 'ramp_offset':
        return x + np.linspace(0.0, value, n)
    if op == 'scale':
        return x * value
    if op == 'ramp_scale':
        return x * np.linspace(1.0, value, n)
    if op == 'set':
        return np.full(n, float(value))
    if op == 'freeze':
        return np.full(n, float(pre))
    if op == 'ramp_to':
        return np.linspace(pre, value, n)
    if op == 'add_noise':
        return x + rng.normal(0.0, value, n)
    raise ValueError(f"Unknown perturbation op '{op}'")


def perturb_block(raw_block: pd.DataFrame, first_row: int, channels, op: str, value,
                  rng: np.random.Generator = None, per_channel_value: dict = None) -> pd.DataFrame:
    """Return a copy of the raw block with `op` applied to `channels` from
    row `first_row` (positional) to the end of the block."""
    out = raw_block.copy()
    for ch in channels:
        col = out.columns.get_loc(ch)
        x = out.iloc[first_row:, col].to_numpy(dtype=float)
        pre = float(out.iloc[first_row - 1, col]) if first_row > 0 else float(x[0])
        v = per_channel_value[ch] if per_channel_value is not None else value
        out.iloc[first_row:, col] = apply_op(op, x, pre, v, rng)
    return out


class BlockScorer:
    """Loads the saved artifacts once and scores raw telemetry blocks."""

    def __init__(self, base_dir='.', raw_csv=None):
        import keras
        self.base = Path(base_dir).resolve()
        self.meta = sg.verify_schema(self.base)
        self.seq_len = int(self.meta['sequence_length'])
        self.stride = int(self.meta['stride'])
        self.warmup = int(self.meta['warmup_rows_removed'])
        self.feature_names = self.meta['feature_names']
        self.fs = float(self.meta['sampling_rate_hz'])
        self.filter_params_path = self.base / 'models' / pc.FILTER_PARAMS_FILENAME
        with open(self.base / 'models' / 'feature_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        self.model = keras.models.load_model(self.base / 'models' / 'lstm_autoencoder.keras')
        self.threshold = load_threshold(self.base)
        self.groups = self.meta.get('redundant_groups', [])

        # Raw telemetry of the TRAINING file (needed to inject into RAW data).
        if raw_csv is None:
            rep = self.base / 'data' / 'processed' / 'preprocessing_report.json'
            if not rep.exists():
                raise FileNotFoundError(f"{rep} not found; pass the training raw CSV explicitly.")
            with open(rep, 'r', encoding='utf-8-sig') as f:
                raw_csv = json.load(f)['input_file']
        raw_path = Path(raw_csv)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw telemetry file not found: {raw_path} (use --raw-csv).")
        self.raw = pd.read_csv(raw_path)
        self.raw_path = raw_path

        # Split boundaries in RAW-row terms (raw row == cleaned row: nothing is dropped in Phase 1)
        self.train_end = self.warmup + int(self.meta['train_rows'])
        self.test_start = self.warmup + int(self.meta['train_rows']) + int(self.meta['val_rows'])
        self.sigma_train = self.raw.iloc[:self.train_end][cc.get_model_input_channels()].std()

        # Per-channel thresholds for attribution (validation errors only).
        X_val = np.load(self.base / 'data' / 'sequences' / 'X_val.npy')
        val_recon = self.model.predict(X_val, batch_size=pc.INFERENCE_BATCH_SIZE, verbose=0)
        val_ch = rollup_to_channels(per_feature_errors(X_val, val_recon), self.feature_names)
        self.channel_thresholds = compute_channel_thresholds(val_ch)

    # ---------------------------------------------------------------- blocks
    def block_bounds(self, n_target_rows: int, fraction: float):
        """(first_row, last_row_exclusive, lookback) of a raw block inside the
        TEST split at relative position `fraction` (0..1): `lookback` real
        rows, then n_target_rows."""
        lookback = int(round(pc.FAULT_INJECTION_LOOKBACK_SECONDS * self.fs))
        n_test = len(self.raw) - self.test_start
        target_start = self.test_start + int(fraction * n_test)
        target_start = min(target_start, len(self.raw) - n_target_rows)
        return target_start - lookback, target_start + n_target_rows, lookback

    def scaled_features(self, raw_block: pd.DataFrame, filters: bool = True) -> np.ndarray:
        """Raw block -> Phase 1 (saved filters, or bypassed) -> Phase 2 -> scaled features."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw_block.to_csv(tmp / 'block.csv', index=False)
            fp_path = self.filter_params_path
            if not filters:
                with open(fp_path, 'r', encoding='utf-8-sig') as f:
                    fp = json.load(f)
                fp['channels'] = {ch: {'type': 'none', 'source': 'bypass'} for ch in fp['channels']}
                fp_path = tmp / 'bypass_filters.json'
                with open(fp_path, 'w') as f:
                    json.dump(fp, f)
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink):
                ok = TelemetryPreprocessor(str(tmp / 'block.csv'), filter_params_path=str(fp_path),
                                           fit_filters=False).run(str(tmp / 'proc'))
                if not ok:
                    raise RuntimeError('Phase 1 failed on block:\n' + sink.getvalue()[-800:])
                feats = TelemetryFeatureExtractor().extract_features(str(tmp / 'proc' / 'cleaned_telemetry.csv'))
            feats = feats.iloc[self.warmup:].reset_index(drop=True)
            return self.scaler.transform(feats[self.feature_names].to_numpy())

    def sequences(self, scaled: np.ndarray) -> np.ndarray:
        n = scaled.shape[0]
        starts = np.arange(0, n - self.seq_len + 1, self.stride)
        return scaled[starts[:, None] + np.arange(self.seq_len)[None, :]]

    def score(self, seqs: np.ndarray):
        """-> (overall error per sequence, per-channel error DataFrame)"""
        recon = self.model.predict(seqs, batch_size=pc.INFERENCE_BATCH_SIZE, verbose=0)
        overall = np.mean((seqs - recon) ** 2, axis=(1, 2))
        ch_err = rollup_to_channels(per_feature_errors(seqs, recon), self.feature_names)
        return overall, ch_err

    def expand_with_group(self, channels) -> set:
        """A channel plus every channel in its redundant group."""
        out = set(channels)
        for g in self.groups:
            if out & set(g):
                out |= set(g)
        return out
