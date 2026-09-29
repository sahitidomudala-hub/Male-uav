"""
Synthetic fault-injection validation (config-driven, no channel names or
magnitudes in code).

WHY: there are no labelled real faults, so detection/attribution cannot be
scored against ground truth. This builds KNOWN, labelled test cases by
perturbing real telemetry and checks (a) whether the perturbed window is
flagged and (b) whether attribution points at the perturbed channel(s).

HOW
  1. Blocks of REAL raw telemetry are taken at several locations in the test
     split (INJECTION_BASE_FRACTIONS_OF_TEST): `lookback` seconds of untouched
     data followed by one sequence-length window.
  2. The perturbation is applied to the RAW signal in that last window.
  3. The block goes through the real pipeline with the saved artifacts:
     Phase 1 (loaded filter cutoffs), Phase 2, saved scaler, saved model.
  4. The last sequence (fully inside the perturbed window) is scored against
     the saved threshold, and channels are ranked by attribution.

SCENARIOS
  * AUTO SWEEP: every model_input channel gets a step_offset and a
    ramp_offset of k standard deviations (pipeline_config
    FAULT_INJECTION_SIGMA_MULTIPLES; sigma from TRAINING rows).
  * NAMED PRESETS: pipeline_config FAULT_INJECTION_SCENARIOS (data, not code).
  Expected channels = the perturbed channel(s) plus their redundant-group
  members (channels that carry the same information, from training metadata).

WHAT THIS IS NOT
  Not evidence about real fault rates. The perturbations are simple and
  exaggerated; a real fault can look far messier. It validates the
  detection + attribution MECHANISM. The data is a fault-free simulation, so
  this is the primary evidence of detection ability.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import channel_config as cc
import pipeline_config as pc
from block_scoring import BlockScorer, perturb_block
from attribute_anomalies import attribute_sequence


def build_scenarios(scorer: BlockScorer) -> list:
    scenarios = []
    known = set(cc.get_physical_channels())
    for sc in pc.FAULT_INJECTION_SCENARIOS:
        if not set(sc['channels']) <= known:
            print(f"  (skipping preset '{sc['name']}': channel not in channel_config)")
            continue
        scenarios.append({'kind': 'named', 'name': sc['name'], 'op': sc['op'],
                          'channels': list(sc['channels']), 'value': sc['value']})
    for ch in cc.get_model_input_channels():
        sigma = float(scorer.sigma_train[ch])
        if not np.isfinite(sigma) or sigma == 0:
            continue
        for k in pc.FAULT_INJECTION_SIGMA_MULTIPLES:
            for op in ('step_offset', 'ramp_offset'):
                scenarios.append({'kind': 'auto', 'name': f'auto_{op}_{ch}_{k:g}sd', 'op': op,
                                  'channels': [ch], 'value': k * sigma})
    return scenarios


def main(base_dir: str = '.', raw_csv: str = None):
    scorer = BlockScorer(base_dir, raw_csv)
    scenarios = build_scenarios(scorer)
    print(f"\nRunning {len(scenarios)} scenarios "
          f"({sum(s['kind'] == 'named' for s in scenarios)} named, {sum(s['kind'] == 'auto' for s in scenarios)} auto-sweep) "
          f"at {len(pc.INJECTION_BASE_FRACTIONS_OF_TEST)} locations in the test split...\n")

    rows = []
    for fraction in pc.INJECTION_BASE_FRACTIONS_OF_TEST:
        first, last, lookback = scorer.block_bounds(scorer.seq_len, fraction)
        base_block = scorer.raw.iloc[first:last].reset_index(drop=True)
        base_time = float(base_block['Time'].iloc[lookback])
        base_seqs = scorer.sequences(scorer.scaled_features(base_block))
        base_err, _ = scorer.score(base_seqs)
        base_error = float(base_err[-1])
        print(f"Location {fraction:.2f} (window at t={base_time:.1f} s): unperturbed error {base_error:.6g} vs threshold "
              f"{scorer.threshold:.6g} -> {'FLAGGED (base window itself is anomalous)' if base_error > scorer.threshold else 'not flagged'}")

        for sc in scenarios:
            perturbed = perturb_block(base_block, lookback, sc['channels'], sc['op'], sc['value'])
            # size of the change actually made, in training standard deviations
            delta = max(float(np.max(np.abs(perturbed[ch].to_numpy() - base_block[ch].to_numpy()))
                              / scorer.sigma_train[ch]) for ch in sc['channels'] if ch in scorer.sigma_train.index)
            seqs = scorer.sequences(scorer.scaled_features(perturbed))
            overall, ch_err = scorer.score(seqs)
            err = float(overall[-1])
            ranked = attribute_sequence(ch_err.iloc[-1], scorer.channel_thresholds, top_n=pc.ATTRIBUTION_TOP_N)
            expected = scorer.expand_with_group(sc['channels'])
            top_channels = ranked['channel'].tolist()
            rows.append({
                'scenario': sc['name'], 'kind': sc['kind'], 'op': sc['op'],
                'channels': ', '.join(sc['channels']), 'value': sc['value'],
                'base_fraction': fraction, 'base_time_s': base_time,
                'perturbation_size_sd': delta,
                'vacuous': bool(delta < pc.VACUOUS_PERTURBATION_MIN_SD),
                'expected_channels': ', '.join(sorted(expected)),
                'baseline_error': base_error, 'baseline_flagged': bool(base_error > scorer.threshold),
                'overall_error': err, 'overall_threshold': scorer.threshold,
                'times_over_threshold': err / scorer.threshold,
                'detected': bool(err > scorer.threshold),
                'top1_channel': top_channels[0],
                'top1_in_expected': top_channels[0] in expected,
                'top_channels': ', '.join(top_channels),
                'any_top_in_expected': bool(expected & set(top_channels)),
            })

    df = pd.DataFrame(rows)
    out_path = Path(base_dir) / 'outputs' / 'fault_injection_validation.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    valid = df[~df.vacuous]
    n_vac = int(df.vacuous.sum())
    def pct(x): return f"{100 * x:.0f}%"
    print('=' * 72)
    print(f"Runs: {len(df)} ({n_vac} vacuous perturbations excluded from the rates below)")
    print(f"DETECTED           : {int(valid.detected.sum())}/{len(valid)} ({pct(valid.detected.mean())})")
    for kind, g in valid.groupby('kind'):
        print(f"  {kind:<6}: detected {int(g.detected.sum())}/{len(g)}")
    for op, g in valid.groupby('op'):
        print(f"  op {op:<12}: detected {int(g.detected.sum())}/{len(g)}")
    for frac, g in valid.groupby('base_fraction'):
        print(f"  location {frac:.2f}: detected {int(g.detected.sum())}/{len(g)}")
    det = valid[valid.detected]
    if len(det):
        print(f"ATTRIBUTION (among detected): top-1 in expected {int(det.top1_in_expected.sum())}/{len(det)}, "
              f"any of top-{pc.ATTRIBUTION_TOP_N} {int(det.any_top_in_expected.sum())}/{len(det)}")
    print(f"Base windows that were themselves flagged: {int(df.drop_duplicates('base_fraction').baseline_flagged.sum())}/{df.base_fraction.nunique()}")
    missed = valid[~valid.detected].groupby('scenario').size()
    if len(missed):
        print("NOT DETECTED (scenario: locations missed): " + ', '.join(f"{k}: {v}" for k, v in missed.items()))
    print('=' * 72)
    print(f"\n✓ Saved: {out_path}")
    print("\nReminder: synthetic, exaggerated perturbations validating the detection+attribution")
    print("MECHANISM on a fault-free simulation; not evidence about real fault rates.")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic fault-injection validation')
    parser.add_argument('--base-dir', type=str, default='.')
    parser.add_argument('--raw-csv', type=str, default=None,
                        help='Raw training telemetry (default: read from preprocessing_report.json)')
    args = parser.parse_args()
    main(args.base_dir, args.raw_csv)
