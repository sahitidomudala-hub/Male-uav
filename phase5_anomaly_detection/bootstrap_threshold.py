"""
Phase 5 extension: Bootstrap confidence interval for the anomaly threshold.

WHY THIS EXISTS
    The threshold (95th percentile of validation reconstruction errors) has
    swung noticeably between training runs on the same data (0.1466 -> 0.0905
    -> 0.1260 across three retrains). This script quantifies HOW MUCH that
    number should be expected to move just from finite-sample variability in
    a 724-second recording, so "the threshold changed" can be judged against
    "how much it could reasonably be expected to change" instead of treated
    as evidence of a bug each time.

METHOD
    Standard percentile bootstrap: resample the validation reconstruction
    errors WITH REPLACEMENT (same size as the original), recompute the 95th
    percentile, repeat many times, and report the resulting distribution's
    spread (e.g. 2.5th-97.5th percentile as a 95% CI on the threshold).

    This does NOT retrain anything and does NOT touch the model -- it only
    resamples the already-computed validation reconstruction errors.

HOW TO READ THE RESULT
    A narrow CI (spread) means the 95th-percentile threshold is a fairly
    stable statistic given this validation set's size and error
    distribution. A wide CI means the threshold is genuinely fragile with
    only ~5,400 validation sequences, and any single point-estimate
    threshold (whatever number it happens to be) should be treated as
    approximate, not authoritative, when explaining the anomaly rate.
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import pipeline_config as _pc
import schema_guard as _sg


def bootstrap_percentile_ci(errors: np.ndarray, percentile: float = None,
                             n_bootstrap: int = None, ci: float = None,
                             seed: int = None) -> dict:
    percentile = _pc.THRESHOLD_PERCENTILE if percentile is None else percentile
    n_bootstrap = _pc.BOOTSTRAP_N_RESAMPLES if n_bootstrap is None else n_bootstrap
    ci = _pc.BOOTSTRAP_CI_LEVEL if ci is None else ci
    seed = _pc.BOOTSTRAP_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    n = len(errors)
    boot_thresholds = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        resample = rng.choice(errors, size=n, replace=True)
        boot_thresholds[i] = np.percentile(resample, percentile)

    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct

    return {
        'point_estimate': float(np.percentile(errors, percentile)),
        'bootstrap_mean': float(boot_thresholds.mean()),
        'bootstrap_std': float(boot_thresholds.std()),
        f'ci_{ci:.0f}pct_lower': float(np.percentile(boot_thresholds, lower_pct)),
        f'ci_{ci:.0f}pct_upper': float(np.percentile(boot_thresholds, upper_pct)),
        'n_validation_sequences': int(n),
        'n_bootstrap_resamples': int(n_bootstrap),
        'percentile_used': percentile,
    }


def anomaly_rate_sensitivity(test_errors: np.ndarray, ci_lower: float, ci_upper: float, point: float) -> dict:
    """Shows how much the ANOMALY RATE itself swings across the threshold's
    plausible range -- this is the number that actually matters for
    presenting results, not the threshold value in isolation."""
    return {
        'anomaly_rate_at_ci_lower_threshold_pct': float(100 * np.mean(test_errors > ci_lower)),
        'anomaly_rate_at_point_estimate_pct': float(100 * np.mean(test_errors > point)),
        'anomaly_rate_at_ci_upper_threshold_pct': float(100 * np.mean(test_errors > ci_upper)),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bootstrap CI for the reconstruction-error threshold')
    parser.add_argument('--base-dir', type=str, default='.')
    parser.add_argument('--percentile', type=float, default=_pc.THRESHOLD_PERCENTILE)
    parser.add_argument('--n-bootstrap', type=int, default=_pc.BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()

    base = Path(args.base_dir)
    _sg.verify_schema(base)
    val_errors = pd.read_csv(base / 'outputs' / 'reconstruction_errors_val.csv')['reconstruction_error_mse'].values
    test_errors = pd.read_csv(base / 'outputs' / 'reconstruction_errors_test.csv')['reconstruction_error_mse'].values

    result = bootstrap_percentile_ci(val_errors, percentile=args.percentile, n_bootstrap=args.n_bootstrap)
    print("Threshold bootstrap result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    sensitivity = anomaly_rate_sensitivity(
        test_errors,
        result[[k for k in result if k.endswith('_lower')][0]],
        result[[k for k in result if k.endswith('_upper')][0]],
        result['point_estimate'],
    )
    print("\nHow much the anomaly rate swings across the threshold's plausible range:")
    for k, v in sensitivity.items():
        print(f"  {k}: {v:.2f}")

    import json
    out = {**result, **sensitivity}
    out_path = base / 'outputs' / 'threshold_bootstrap_ci.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved: {out_path}")
