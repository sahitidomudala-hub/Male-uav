"""
Noise-robustness sweep.

QUESTION ANSWERED: how much added sensor noise does the trained pipeline
tolerate before it starts flagging windows, and how much of that tolerance is
thanks to the Phase 1 filters?

METHOD
  * A contiguous block of REAL raw telemetry is taken from the middle of the
    test split (pipeline_config NOISE_SWEEP_BLOCK_LENGTH_SECONDS, preceded by
    FAULT_INJECTION_LOOKBACK_SECONDS of untouched lookback).
  * For each level k (NOISE_SWEEP_LEVELS) zero-mean white Gaussian noise with
    std = k * sigma_train is added to the RAW signal inside the block, where
    sigma_train is that channel's std over the TRAINING rows.
      - mode 'single': one channel at a time (every model_input channel)
      - mode 'all'   : all model_input channels together
  * The block runs through the real pipeline with the saved artifacts (Phase 1
    with the loaded filter cutoffs, Phase 2, saved scaler, saved model).
    Nothing is refit.
  * The same runs are repeated with the Phase 1 filters bypassed (experiment
    flag inside this script only) to show what the filters contribute.
  * Flagged fraction = windows fully/partly inside the noisy block whose error
    exceeds the saved threshold; baseline = the SAME block without noise.
    Compare against the baseline, not against zero.

READ THIS BEFORE QUOTING RESULTS
  * The data is a fault-free simulation; this measures sensor-noise
    sensitivity, not fault detection.
  * Noise is white Gaussian; real sensor noise can be coloured or spiky.
  * The percentile threshold flags a share of clean windows by construction,
    hence the baseline column.
  * Flagging heavy noise means "sensor degradation / poor signal quality",
    not an engine fault.
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


def main(base_dir: str = '.', raw_csv: str = None):
    scorer = BlockScorer(base_dir, raw_csv)
    block_rows = int(round(pc.NOISE_SWEEP_BLOCK_LENGTH_SECONDS * scorer.fs))
    fractions = pc.INJECTION_BASE_FRACTIONS_OF_TEST
    first, last, lookback = scorer.block_bounds(block_rows, fractions[len(fractions) // 2])
    base_block = scorer.raw.iloc[first:last].reset_index(drop=True)
    channels = cc.get_model_input_channels()
    print(f"Block: raw rows {first}..{last - 1} (t={base_block['Time'].iloc[0]:.1f}-{base_block['Time'].iloc[-1]:.1f} s); "
          f"noise applied to the last {block_rows} rows ({pc.NOISE_SWEEP_BLOCK_LENGTH_SECONDS} s).")

    # sequences whose window overlaps the noisy region: those ending at or after its first row
    def flagged_fraction(errors, n_rows_after_warmup):
        # sequence j covers feature rows [j, j+L-1]; feature row r <-> block row r + warmup
        j = np.arange(len(errors))
        ends_block_row = j + scorer.seq_len - 1 + scorer.warmup
        inside = ends_block_row >= lookback
        return float(np.mean(errors[inside] > scorer.threshold)), int(inside.sum())

    results = []
    baselines = {}
    for filters_on in (True, False):
        seqs = scorer.sequences(scorer.scaled_features(base_block, filters=filters_on))
        err, _ = scorer.score(seqs)
        baselines[filters_on], n_seq = flagged_fraction(err, len(err))
        print(f"Baseline (no noise, filters {'ON' if filters_on else 'BYPASSED'}): "
              f"{baselines[filters_on] * 100:.1f}% of {n_seq} windows flagged")

    rng_seed = pc.NOISE_SWEEP_SEED
    modes = [('single', [ch]) for ch in channels] + [('all', channels)]
    total = len(modes) * len(pc.NOISE_SWEEP_LEVELS) * 2
    done = 0
    for mode, chs in modes:
        for k in pc.NOISE_SWEEP_LEVELS:
            per_channel_sigma = {ch: k * float(scorer.sigma_train[ch]) for ch in chs}
            rng = np.random.default_rng(rng_seed)
            noisy = perturb_block(base_block, lookback, chs, 'add_noise', None, rng=rng,
                                  per_channel_value=per_channel_sigma)
            for filters_on in (True, False):
                seqs = scorer.sequences(scorer.scaled_features(noisy, filters=filters_on))
                err, ch_err = scorer.score(seqs)
                frac, n_seq = flagged_fraction(err, len(err))
                # attribution: channel with the largest error/threshold ratio, averaged over the noisy windows
                ratio = (ch_err / scorer.channel_thresholds).iloc[-n_seq:].mean()
                top1 = ratio.idxmax()
                results.append({
                    'mode': mode, 'channel': chs[0] if mode == 'single' else 'ALL',
                    'level_k': k, 'filters_on': filters_on,
                    'flagged_fraction_noisy': frac,
                    'flagged_fraction_baseline': baselines[filters_on],
                    'excess_over_baseline': frac - baselines[filters_on],
                    'top1_channel': top1,
                    'top1_is_noisy_channel': (top1 in chs) if mode == 'single' else None,
                    'noise_sigma_abs': per_channel_sigma[chs[0]] if mode == 'single' else None,
                })
                done += 1
        print(f"  finished mode '{mode}' {chs[0] if mode == 'single' else '(all channels)'} [{done}/{total}]")

    df = pd.DataFrame(results)
    out = Path(base_dir) / 'outputs' / 'noise_sweep.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    margin = pc.NOISE_SWEEP_FLAGGED_MARGIN
    print('\n' + '=' * 78)
    print(f"Smallest noise level k (in units of the channel's training std) at which the flagged "
          f"fraction exceeds its baseline by more than {margin:.2f}:")
    print(f"{'channel':<12}{'filters ON':>14}{'filters OFF':>14}")
    for ch in [c for c in df.channel.unique()]:
        cells = []
        for fo in (True, False):
            g = df[(df.channel == ch) & (df.filters_on == fo) & (df.excess_over_baseline > margin)]
            cells.append(f"{g.level_k.min():g}" if len(g) else f">{max(pc.NOISE_SWEEP_LEVELS):g}")
        print(f"{ch:<12}{cells[0]:>14}{cells[1]:>14}")
    print('=' * 78)
    print(f"\n✓ Saved: {out}")
    print("\nReminder: white Gaussian noise on a fault-free simulation; measures noise sensitivity only.")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Noise robustness sweep (no retraining)')
    parser.add_argument('--base-dir', type=str, default='.')
    parser.add_argument('--raw-csv', type=str, default=None)
    args = parser.parse_args()
    main(args.base_dir, args.raw_csv)
