"""
Phase 1: Preprocessing for Engine Telemetry (config-driven)

Everything channel-specific comes from channel_config.py and every numeric
parameter from pipeline_config.py -- nothing about specific channels, cutoffs
or limits is hardcoded here.

Order of operations (per channel):
    raw -> noise estimate (report only) -> causal filter -> correction rule
    -> cleaned

WHY FILTER BEFORE CORRECTING: if sensor noise straddles a physical limit
(e.g. FuelFlow noise around 0), clipping the raw signal first is a half-wave
rectification that biases the average upward. Filtering first lets the noise
average out before any value is forced to a boundary.

FILTER CUTOFFS -- FIT vs APPLY
    A Butterworth channel declared ('butterworth', 'auto') gets its cutoff
    from the channel's own bandwidth on the TRAINING rows (first TRAIN_RATIO
    of the file) -- see pipeline_config AUTO_CUTOFF_*.
      * fit_filters=True  (training / detect modes): derive the cutoffs and
        save them to models/filter_params.json.
      * fit_filters=False (inference): LOAD the saved cutoffs. Nothing is
        re-derived from the incoming file, exactly like the scaler.

CAUSALITY: Butterworth uses scipy.signal.lfilter (forward only) and median
filters use a trailing rolling window. Never filtfilt (needs future samples).
Filters start at steady state for the first sample (pipeline_config
FILTER_STEADY_STATE_INIT) so large DC offsets do not create a fake start-up
transient; this uses only the first sample and is therefore still causal.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi, welch

# channel_config.py / pipeline_config.py live at the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import channel_config as _cc
import pipeline_config as _pc

TIME_COLUMN = _cc.TIME_COLUMN

# Correction-rule registry. A rule is (name, params); to support a new kind of
# correction add one entry here and reference it by name in channel_config.
CORRECTION_RULES = {
    'clip_min': lambda s, p: s.clip(lower=p['min']),
    'clip_max': lambda s, p: s.clip(upper=p['max']),
    'clip_range': lambda s, p: s.clip(lower=p['min'], upper=p['max']),
}


def default_filter_params_path() -> Path:
    return REPO_ROOT / 'models' / _pc.FILTER_PARAMS_FILENAME


class TelemetryPreprocessor:
    """Quality-aware, config-driven telemetry validation, filtering and repair."""

    def __init__(self, input_csv_path: str,
                 channel_config: Optional[dict] = None,
                 filter_params_path: Optional[str] = None,
                 fit_filters: bool = True):
        self.input_path = Path(input_csv_path)
        self.channel_config = channel_config if channel_config is not None else _cc.CHANNEL_CONFIG
        self.filter_params_path = Path(filter_params_path) if filter_params_path else default_filter_params_path()
        self.fit_filters = fit_filters

        self.df_raw = None
        self.df_filtered = None
        self.df_cleaned = None
        self.quality_flags = None
        self.filter_params = None
        self.stats = {}

        self.channels = list(self.channel_config.keys())
        self.expected_columns = [TIME_COLUMN] + self.channels
        self.corrected_channels = [ch for ch, cfg in self.channel_config.items() if cfg.get('correction')]
        self.unchanged_channels = [ch for ch in self.channels if ch not in self.corrected_channels]

        for ch in self.corrected_channels:
            rule = self.channel_config[ch]['correction'][0]
            if rule not in CORRECTION_RULES:
                raise ValueError(f"Channel '{ch}' uses unknown correction rule '{rule}'. "
                                 f"Known rules: {sorted(CORRECTION_RULES)}")

    # ------------------------------------------------------------------ load
    def load_data(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        print(f"Loading data from: {self.input_path}")
        self.df_raw = pd.read_csv(self.input_path)
        print(f"  Loaded: {len(self.df_raw)} rows × {len(self.df_raw.columns)} columns")
        self.stats['input_file'] = str(self.input_path)
        self.stats['total_rows'] = len(self.df_raw)
        self.stats['total_columns'] = len(self.df_raw.columns)
        self.stats['columns'] = list(self.df_raw.columns)

    def validate_schema(self) -> None:
        print("\nValidating schema...")
        missing = set(self.expected_columns) - set(self.df_raw.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        extra = [c for c in self.df_raw.columns if c not in self.expected_columns]
        if extra:
            print(f"  ⚠ Ignoring {len(extra)} column(s) not in channel_config: {extra}")
        print(f"  ✓ All {len(self.expected_columns)} required columns present")
        self.stats['schema_valid'] = True
        self.stats['ignored_columns'] = extra

    def validate_timestamps(self) -> None:
        print("\nValidating timestamps...")
        time = self.df_raw[TIME_COLUMN]
        time_diffs = time.diff()[1:]
        if (time_diffs <= 0).any():
            print("  ⚠ WARNING: Time is not strictly monotonically increasing")
            self.stats['time_monotonic'] = False
        else:
            print("  ✓ Time is monotonically increasing")
            self.stats['time_monotonic'] = True

        median_interval = time_diffs.median()
        sampling_rate = 1.0 / median_interval if median_interval > 0 else 0
        print(f"  Sampling interval (median): {median_interval:.6f} s")
        print(f"  Sampling rate (approx): {sampling_rate:.2f} Hz")
        print(f"  Interval std dev: {time_diffs.std():.6f} s")
        print(f"  Duration: {time.max():.2f} s")

        self.stats['sampling_interval_median_s'] = float(median_interval)
        self.stats['sampling_interval_mean_s'] = float(time_diffs.mean())
        self.stats['sampling_interval_std_s'] = float(time_diffs.std())
        self.stats['sampling_interval_min_s'] = float(time_diffs.min())
        self.stats['sampling_interval_max_s'] = float(time_diffs.max())
        self.stats['sampling_rate_hz'] = float(sampling_rate)
        self.stats['duration_s'] = float(time.max())

    def detect_missing_and_infinite(self) -> None:
        print("\nDetecting missing and infinite values...")
        nan_by_col, inf_by_col = {}, {}
        for col in self.channels:
            nan_count = int(self.df_raw[col].isna().sum())
            inf_count = int(np.isinf(self.df_raw[col]).sum())
            if nan_count:
                nan_by_col[col] = nan_count
            if inf_count:
                inf_by_col[col] = inf_count
        total_nan, total_inf = sum(nan_by_col.values()), sum(inf_by_col.values())
        if total_nan == 0:
            print("  ✓ No missing (NaN) values detected")
        else:
            print(f"  ⚠ Found {total_nan} missing values: {nan_by_col}")
        if total_inf == 0:
            print("  ✓ No infinite values detected")
        else:
            print(f"  ⚠ Found {total_inf} infinite values: {inf_by_col}")
        self.stats['total_nan_values'] = total_nan
        self.stats['total_inf_values'] = total_inf
        self.stats['nan_by_column'] = nan_by_col

    # ------------------------------------------------------- noise estimate
    def estimate_noise(self) -> None:
        """
        Robust per-channel high-frequency noise estimate on the RAW signal:
        sigma = MAD(second difference) * NOISE_MAD_SCALE / sqrt(6).
        Reported so downstream layers can tell a noisy input from a real
        anomaly. Median-based, so isolated steps/spikes barely affect it.
        """
        print("\nEstimating per-channel noise level (raw signal)...")
        estimates = {}
        for ch in self.channels:
            x = self.df_raw[ch].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if len(x) < 3:
                continue
            d2 = np.diff(x, n=2)
            mad = np.median(np.abs(d2 - np.median(d2)))
            sigma = float(_pc.NOISE_MAD_SCALE * mad / np.sqrt(6.0))
            signal_std = float(np.std(x))
            estimates[ch] = {
                'sigma_hf': sigma,
                'relative_to_signal_std': float(sigma / signal_std) if signal_std > 0 else None,
            }
        self.stats['noise_estimate'] = estimates
        worst = sorted(((v['relative_to_signal_std'] or 0.0, ch) for ch, v in estimates.items()), reverse=True)[:3]
        print("  Highest relative noise: " + ", ".join(f"{ch}={r:.2e}" for r, ch in worst))

    # ------------------------------------------------------------- filtering
    def _derive_filter_params(self, fs: float) -> dict:
        n_fit = int(len(self.df_raw) * _pc.TRAIN_RATIO)
        nyquist = fs / 2.0
        max_hz = min(_pc.AUTO_CUTOFF_MAX_HZ, _pc.NYQUIST_GUARD * nyquist)
        channels = {}
        for ch, (ftype, param) in _cc.get_filter_config(self.channel_config).items():
            if ftype == 'butterworth' and param == 'auto':
                x = self.df_raw[ch].to_numpy(dtype=float)[:n_fit]
                x = x - np.nanmean(x)
                if not np.isfinite(x).all() or np.std(x) == 0:
                    cutoff, source = _pc.AUTO_CUTOFF_FALLBACK_HZ, 'fallback'
                else:
                    f, p = welch(x, fs=fs, nperseg=min(_pc.AUTO_CUTOFF_WELCH_NPERSEG, len(x)))
                    cum = np.cumsum(p) / p.sum()
                    f_bw = float(f[min(np.searchsorted(cum, _pc.AUTO_CUTOFF_POWER_FRACTION), len(f) - 1)])
                    cutoff, source = float(np.clip(f_bw, _pc.AUTO_CUTOFF_MIN_HZ, max_hz)), 'auto'
                channels[ch] = {'type': 'butterworth', 'cutoff_hz': cutoff, 'source': source}
            elif ftype == 'butterworth':
                channels[ch] = {'type': 'butterworth', 'cutoff_hz': float(param), 'source': 'fixed'}
            elif ftype == 'median':
                channels[ch] = {'type': 'median', 'window': int(param), 'source': 'fixed'}
            elif ftype == 'none':
                channels[ch] = {'type': 'none', 'source': 'fixed'}
            else:
                raise ValueError(f"Unknown filter type '{ftype}' for channel '{ch}'")
        return {
            'fs_hz': float(fs),
            'fit_rows': int(n_fit),
            'fit_fraction': _pc.TRAIN_RATIO,
            'power_fraction': _pc.AUTO_CUTOFF_POWER_FRACTION,
            'min_cutoff_hz': _pc.AUTO_CUTOFF_MIN_HZ,
            'max_cutoff_hz': float(max_hz),
            'channels': channels,
        }

    def resolve_filter_params(self) -> None:
        fs = self.stats['sampling_rate_hz']
        if self.fit_filters:
            self.filter_params = self._derive_filter_params(fs)
            self.filter_params_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filter_params_path, 'w') as f:
                json.dump(self.filter_params, f, indent=2)
            print(f"\nFilter parameters FITTED on the first {self.filter_params['fit_rows']} rows "
                  f"and saved: {self.filter_params_path}")
        else:
            if not self.filter_params_path.exists():
                raise FileNotFoundError(
                    f"Saved filter parameters not found: {self.filter_params_path}. "
                    "They are produced when the pipeline is trained; run training first.")
            with open(self.filter_params_path, 'r', encoding='utf-8-sig') as f:
                self.filter_params = json.load(f)
            saved_fs = self.filter_params['fs_hz']
            if abs(fs - saved_fs) > _pc.SAMPLING_RATE_TOLERANCE_FRACTION * saved_fs:
                raise ValueError(f"Input sampling rate {fs:.3f} Hz differs from the training rate "
                                 f"{saved_fs:.3f} Hz beyond tolerance; the model and windows are "
                                 "defined in samples, so this file cannot be processed as-is.")
            missing = [ch for ch in self.channels if ch not in self.filter_params['channels']]
            if missing:
                raise ValueError(f"Saved filter parameters lack channels {missing}; "
                                 "the channel config changed -- retrain.")
            print(f"\nFilter parameters LOADED (not re-derived): {self.filter_params_path}")

    def apply_channel_filters(self) -> None:
        print("\nApplying per-channel causal noise filters...")
        fs = self.stats.get('sampling_rate_hz')
        if not fs or fs <= 0:
            raise ValueError("Sampling rate not available -- validate_timestamps() must run first.")
        if self.filter_params is None:
            self.resolve_filter_params()
        nyquist = fs / 2.0

        self.df_filtered = self.df_raw[[TIME_COLUMN] + self.channels].copy()
        applied = {}
        for ch in self.channels:
            spec = self.filter_params['channels'][ch]
            x = self.df_filtered[ch].to_numpy(dtype=float)
            if spec['type'] == 'butterworth':
                wn = min(spec['cutoff_hz'] / nyquist, _pc.NYQUIST_GUARD)
                b, a = butter(N=_pc.BUTTERWORTH_ORDER, Wn=wn, btype='low')
                if _pc.FILTER_STEADY_STATE_INIT:
                    finite = np.isfinite(x)
                    x0 = x[finite][0] if finite.any() else 0.0
                    y, _ = lfilter(b, a, x, zi=lfilter_zi(b, a) * x0)
                else:
                    y = lfilter(b, a, x)
                applied[ch] = f"butterworth low-pass, {spec['cutoff_hz']:.3f} Hz cutoff ({spec['source']}, fs={fs:.2f} Hz)"
            elif spec['type'] == 'median':
                y = (pd.Series(x).rolling(window=spec['window'], min_periods=_pc.ROLLING_MEDIAN_MIN_PERIODS,
                                          center=False).median().to_numpy())
                applied[ch] = f"causal median filter, window={spec['window']} samples"
            elif spec['type'] == 'none':
                y = x
                applied[ch] = "no filter"
            else:
                raise ValueError(f"Unknown filter type '{spec['type']}' for channel '{ch}'")
            self.df_filtered[ch] = y

        for ch, desc in applied.items():
            print(f"  {ch}: {desc}")
        self.stats['filters_applied'] = applied
        self.stats['filter_params'] = self.filter_params
        self.stats['filter_sampling_rate_hz'] = float(fs)
        print(f"  ✓ Filtering complete ({len(applied)} channels)")

    # ----------------------------------------------- corrections / quality
    def detect_and_repair(self) -> None:
        """Generic correction: apply each channel's rule to the FILTERED signal.
        A sample is flagged 'invalid/corrected' iff the rule moved its
        value by more than the numerical-noise tolerance (values are clipped
        regardless)."""
        print("\nApplying correction rules to filtered signals...")
        self.df_cleaned = self.df_filtered.copy()
        self.quality_flags = pd.DataFrame(index=self.df_raw.index)
        self.quality_flags['timestamp'] = self.df_raw[TIME_COLUMN]
        n = len(self.df_raw)
        corrections = {}

        for ch in self.channels:
            rule_cfg = self.channel_config[ch].get('correction')
            if not rule_cfg:
                self.quality_flags[f'{ch}_quality'] = 0
                continue
            rule, params = rule_cfg
            filtered = self.df_filtered[ch]
            raw = self.df_raw[ch]
            clipped = CORRECTION_RULES[rule](filtered, params)
            # Always clip; only FLAG violations larger than numerical noise
            # (fraction of this channel's dynamic range, see pipeline_config).
            dyn_range = float(filtered.max() - filtered.min())
            tol = _pc.CORRECTION_TOLERANCE_FRACTION_OF_RANGE * dyn_range
            invalid = ((clipped - filtered).abs() > tol) & filtered.notna()
            self.df_cleaned[ch] = clipped
            self.quality_flags[f'{ch}_quality'] = invalid.astype(int)

            count = int(invalid.sum())
            print(f"  {ch}: rule={rule} {params} -> corrected {count} samples ({count / n * 100:.2f}%)")
            print(f"    raw range [{raw.min():.6g}, {raw.max():.6g}], "
                  f"filtered range [{filtered.min():.6g}, {filtered.max():.6g}]")
            p = ch.lower()
            self.stats.update({
                f'{p}_raw_min': float(raw.min()), f'{p}_raw_max': float(raw.max()),
                f'{p}_raw_mean': float(raw.mean()), f'{p}_raw_std': float(raw.std()),
                f'{p}_invalid_count': count, f'{p}_invalid_percent': float(count / n * 100),
                f'{p}_cleaned_min': float(clipped.min()), f'{p}_cleaned_max': float(clipped.max()),
                f'{p}_cleaned_mean': float(clipped.mean()), f'{p}_cleaned_std': float(clipped.std()),
                f'{p}_corrected_count': count,
            })
            corrections[ch] = {'rule': rule, 'params': params, 'corrected_count': count,
                               'corrected_percent': float(count / n * 100)}
        self.stats['corrections'] = corrections
        self.stats['corrected_channels'] = self.corrected_channels

        # Channels without a rule must have gone raw -> filtered -> cleaned only.
        unchanged_ok = True
        for col in self.unchanged_channels:
            if not (self.df_filtered[col] == self.df_cleaned[col]).all():
                print(f"  ⚠ WARNING: {col} was modified beyond filtering!")
                unchanged_ok = False
        if unchanged_ok:
            print(f"  ✓ {len(self.unchanged_channels)} channels without a correction rule: filtered only")
        self.stats['unchanged_channels_verified'] = unchanged_ok
        self.stats['unchanged_channels'] = self.unchanged_channels
        print(f"  Created quality flags for {len(self.quality_flags.columns) - 1} channels")

    # ----------------------------------------------------------------- save
    def save_outputs(self, output_dir: str = 'data/processed') -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving outputs to {output_path}...")

        cleaned_csv = output_path / 'cleaned_telemetry.csv'
        self.df_cleaned.to_csv(cleaned_csv, index=False)
        print(f"  ✓ Cleaned telemetry: {cleaned_csv}")

        quality_csv = output_path / 'quality_flags.csv'
        self.quality_flags.to_csv(quality_csv, index=False)
        print(f"  ✓ Quality flags: {quality_csv}")

        report_json = output_path / 'preprocessing_report.json'
        with open(report_json, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"  ✓ Statistics report: {report_json}")

    def run(self, output_dir: str = 'data/processed') -> bool:
        try:
            print("=" * 80)
            print(" " * 25 + "PHASE 1 - PREPROCESSING")
            print("=" * 80)

            self.load_data()
            self.validate_schema()
            self.validate_timestamps()
            self.detect_missing_and_infinite()
            self.estimate_noise()
            self.resolve_filter_params()
            self.apply_channel_filters()
            self.detect_and_repair()
            self.save_outputs(output_dir)

            print("\n" + "=" * 80)
            print(" " * 28 + "PHASE 1 COMPLETE")
            print("=" * 80)
            print(f"\nProcessed {len(self.df_raw)} rows")
            for ch in self.corrected_channels:
                print(f"  {ch} corrected: {self.stats[f'{ch.lower()}_corrected_count']} samples")
            print(f"  Channels without correction rule: {len(self.unchanged_channels)}")
            return True

        except Exception as e:
            print(f"\n{'=' * 80}")
            print(" " * 28 + "PHASE 1 FAILED")
            print(f"{'=' * 80}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 1 preprocessing')
    parser.add_argument('input_csv')
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--apply-filters', action='store_true',
                        help='LOAD saved filter cutoffs (inference) instead of deriving them from this file')
    args = parser.parse_args()

    ok = TelemetryPreprocessor(args.input_csv, fit_filters=not args.apply_filters).run(args.output_dir)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
