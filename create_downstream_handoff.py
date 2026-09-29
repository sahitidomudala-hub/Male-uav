"""
Downstream health-vector handoff (candidate schema, NOT a mandated one).

Maps each scored sequence back to the telemetry it covers and attaches the
per-channel MEAN of the cleaned telemetry over that window. Everything that
defines the mapping is read from the training metadata (sequence length,
stride, number of warm-up rows dropped) -- nothing is hardcoded here:

    sequence i  ->  cleaned-telemetry rows [ start .. start + L - 1 ]
    start = warmup_rows_removed + i * stride,  L = sequence_length

Channel columns come from channel_config (ALL channels, including the ones
that are passthrough-only for the model). Units are written next to the CSV
as <name>_units.json, generated from channel_config.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import channel_config as cc
import schema_guard as sg


def write_handoff_units(path: Path) -> None:
    doc = {
        'time_column': {'name': cc.TIME_COLUMN, 'unit': 's', 'meaning': 'mission-elapsed seconds'},
        'channel_mean_columns': {f'{ch}_mean': {'source_channel': ch, 'unit': unit}
                                 for ch, unit in cc.get_units_dict().items()},
        'passthrough_channels_not_used_by_model': cc.get_passthrough_channels(),
        'note': 'Candidate integration schema; not mandated by the project documents. '
                '*_mean = mean of the cleaned (filtered + corrected) channel over the sequence window.',
    }
    with open(path, 'w') as f:
        json.dump(doc, f, indent=2)


def create_downstream_handoff(inference_results_path: str, cleaned_telemetry_path: str,
                              features_path: str, output_path: str, meta: dict = None) -> pd.DataFrame:
    meta = sg.load_metadata(ROOT) if meta is None else meta
    seq_len = int(meta['sequence_length'])
    stride = int(meta['stride'])
    warmup = int(meta['warmup_rows_removed'])

    inference_df = pd.read_csv(inference_results_path)
    cleaned_df = pd.read_csv(cleaned_telemetry_path)
    features_df = pd.read_csv(features_path)
    channels = cc.get_physical_channels()

    usable_rows = len(features_df) - warmup
    expected_sequences = (usable_rows - seq_len) // stride + 1
    if expected_sequences != len(inference_df):
        raise ValueError(f"Sequence count mismatch: {len(inference_df)} scored, but "
                         f"{expected_sequences} expected from {usable_rows} usable rows.")
    if len(cleaned_df) != len(features_df):
        raise ValueError("cleaned telemetry and features have different row counts.")

    print(f"Creating downstream handoff for {len(inference_df)} sequences "
          f"(length {seq_len}, stride {stride}, warm-up rows {warmup})...")

    seq_idx = inference_df['sequence_index'].to_numpy(dtype=int)
    start = warmup + seq_idx * stride
    end = start + seq_len - 1
    if end.max() >= len(cleaned_df):
        raise ValueError("A sequence extends beyond the cleaned telemetry.")

    time = cleaned_df[cc.TIME_COLUMN].to_numpy()
    rolling_means = cleaned_df[channels].rolling(window=seq_len).mean()

    out = pd.DataFrame({
        'sequence_index': seq_idx,
        'start_time': time[start],
        'end_time': time[end],
        'reconstruction_error': inference_df['reconstruction_error'].to_numpy(),
        'threshold': inference_df['threshold'].to_numpy(),
        'classification': inference_df['classification'].to_numpy(),
    })
    means = rolling_means.iloc[end].reset_index(drop=True)
    for ch in channels:
        out[f'{ch}_mean'] = means[ch].to_numpy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    write_handoff_units(output_path.with_name(output_path.stem + '_units.json'))
    print(f"  ✓ Created {len(out)} handoff records -> {output_path}")
    return out


if __name__ == '__main__':
    create_downstream_handoff('outputs/inference_results.csv', 'data/inference/processed/cleaned_telemetry.csv',
                              'data/inference/features/extracted_features.csv',
                              'outputs/downstream_health_vector.csv')
