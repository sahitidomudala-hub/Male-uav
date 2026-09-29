"""
Phase 1 validation (config-driven).

Checks the Phase 1 outputs against the SPEC in channel_config / pipeline_config
and the SAVED filter parameters -- nothing about specific channels is
hardcoded, so it stays correct when the channel table changes.

Checks
  1. Output files exist
  2. Row counts preserved; quality flags aligned; no rows removed
  3. Column structure: cleaned = [Time] + configured channels;
     quality flags = timestamp + one <channel>_quality per channel
  4. Every channel WITH a correction rule: cleaned values lie inside the rule's
     limits and equal rule(filtered); flags mark exactly the samples the rule
     moved by more than the numerical-noise tolerance
  5. Every channel WITHOUT a rule equals its expected filtered signal
     (filtered only, nothing else applied)
  6. No NaN/Inf introduced
  7. Statistics report contains the required entries
The expected filtered signal is recomputed by re-running the production
filtering code with the saved cutoffs (models/filter_params.json).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))
from preprocess import TelemetryPreprocessor, CORRECTION_RULES
import channel_config as cc
import pipeline_config as pc


def validate_phase1(raw_csv_path: str, processed_dir: str = 'data/processed') -> bool:
    print("=" * 80)
    print(" " * 23 + "PHASE 1 - VALIDATION")
    print("=" * 80)
    proc = Path(processed_dir)
    cleaned_csv, quality_csv, report_json = (proc / 'cleaned_telemetry.csv',
                                             proc / 'quality_flags.csv', proc / 'preprocessing_report.json')
    ok_all = True

    def check(cond, pass_msg, fail_msg):
        nonlocal ok_all
        print(f"  {'✓ PASS' if cond else '✗ FAIL'}: {pass_msg if cond else fail_msg}")
        ok_all &= bool(cond)
        return bool(cond)

    print("\n[1/7] Checking output files exist...")
    for p in (cleaned_csv, quality_csv, report_json):
        check(p.exists(), f"{p} exists", f"{p} not found")
    if not ok_all:
        print("\nValidation aborted: missing output files")
        return False

    df_raw = pd.read_csv(raw_csv_path)
    df_clean = pd.read_csv(cleaned_csv)
    df_q = pd.read_csv(quality_csv)
    with open(report_json, 'r', encoding='utf-8-sig') as f:
        report = json.load(f)
    channels = cc.get_physical_channels()

    print("\n[2/7] Row counts...")
    check(len(df_raw) == len(df_clean) == len(df_q),
          f"raw, cleaned and quality flags all have {len(df_raw)} rows",
          f"row counts differ: raw {len(df_raw)}, cleaned {len(df_clean)}, quality {len(df_q)}")

    print("\n[3/7] Column structure...")
    check(list(df_clean.columns) == [cc.TIME_COLUMN] + channels,
          f"cleaned columns = Time + {len(channels)} configured channels",
          f"cleaned columns differ from channel_config: {list(df_clean.columns)}")
    check(list(df_q.columns) == ['timestamp'] + [f'{c}_quality' for c in channels],
          "quality flag columns = timestamp + one per channel", f"quality columns unexpected: {list(df_q.columns)}")

    # Expected filtered signal via the production code + SAVED cutoffs.
    ref = TelemetryPreprocessor(raw_csv_path, fit_filters=False)
    ref.load_data()
    ref.validate_timestamps()
    ref.apply_channel_filters()
    filt = ref.df_filtered

    print("\n[4/7] Channels with a correction rule...")
    for ch, (rule, params) in cc.get_channels_with_correction().items():
        clipped = CORRECTION_RULES[rule](filt[ch], params)
        lo, hi = params.get('min', -np.inf), params.get('max', np.inf)
        inside = df_clean[ch].between(lo, hi).all()
        check(inside, f"{ch}: all cleaned values within [{lo}, {hi}]",
              f"{ch}: cleaned values outside [{lo}, {hi}] (min {df_clean[ch].min()}, max {df_clean[ch].max()})")
        check(np.allclose(df_clean[ch].to_numpy(), clipped.to_numpy(), rtol=1e-9, atol=1e-12),
              f"{ch}: cleaned == {rule}(filtered)", f"{ch}: cleaned does not equal {rule}(filtered)")
        tol = pc.CORRECTION_TOLERANCE_FRACTION_OF_RANGE * float(filt[ch].max() - filt[ch].min())
        expected_flags = ((clipped - filt[ch]).abs() > tol).astype(int)
        check((df_q[f'{ch}_quality'].to_numpy() == expected_flags.to_numpy()).all(),
              f"{ch}: quality flags mark exactly the {int(expected_flags.sum())} samples the rule moved beyond tolerance",
              f"{ch}: quality flags disagree with the samples the rule moved")

    print("\n[5/7] Channels without a correction rule (filtered only)...")
    plain = [c for c in channels if c not in cc.get_channels_with_correction()]
    bad = [c for c in plain if not np.allclose(filt[c].to_numpy(), df_clean[c].to_numpy(), rtol=1e-9, atol=1e-12)]
    check(not bad, f"all {len(plain)} channels match their expected filtered signal",
          f"channels modified beyond filtering: {bad}")
    check(all((df_q[f'{c}_quality'] == 0).all() for c in plain), "no quality flags raised for them",
          "quality flags raised for a channel without a rule")

    print("\n[6/7] NaN / Inf...")
    num = df_clean.select_dtypes('number')
    check(not num.isna().any().any() and not np.isinf(num).any().any(), "no NaN/Inf in cleaned telemetry",
          "NaN or Inf found in cleaned telemetry")

    print("\n[7/7] Statistics report...")
    required = ['sampling_rate_hz', 'filters_applied', 'noise_estimate', 'corrections', 'unchanged_channels_verified']
    missing = [k for k in required if k not in report]
    check(not missing, "all required entries present", f"missing entries: {missing}")

    print("\n" + "=" * 80)
    print(" " * 25 + ("ALL VALIDATIONS PASSED" if ok_all else "SOME VALIDATIONS FAILED"))
    print("=" * 80)
    return ok_all


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate.py <raw_csv_path> [processed_dir]")
        sys.exit(1)
    sys.exit(0 if validate_phase1(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'data/processed') else 1)


if __name__ == '__main__':
    main()
