"""
inference_core.py
==================
The ONE implementation of inference (used by run_pipeline.py --mode inference
and phase5_anomaly_detection/run_inference.py). Nothing is refit, retrained
or recalculated:

    scaler      LOADED  models/feature_scaler.pkl
    filters     LOADED  models/filter_params.json   (derived at training time)
    model       LOADED  models/lstm_autoencoder.keras
    threshold   LOADED  outputs/anomaly_detection_report.json

Everything tied to the trained model (feature names/order, sequence length,
stride, sampling rate, warm-up rows) is READ from data/sequences/metadata.json.
schema_guard.verify_schema() refuses to run if the configuration no longer
matches the trained artifacts.
"""

import os
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for _sub in ('phase1_preprocessing', 'phase2_feature_extraction'):
    sys.path.insert(0, str(ROOT / _sub))
sys.path.insert(0, str(ROOT))

import pipeline_config as pc
import schema_guard as sg
from preprocess import TelemetryPreprocessor
from extract_features import run_phase2
from create_downstream_handoff import create_downstream_handoff


def score_scaled(model, scaled: np.ndarray, seq_len: int, stride: int,
                 batch_size: int = None, chunk: int = None) -> np.ndarray:
    """Reconstruction error (MSE over timesteps and features) of every window
    of an already-scaled feature array. Windows are built chunk by chunk."""
    batch_size = pc.INFERENCE_BATCH_SIZE if batch_size is None else batch_size
    chunk = pc.INFERENCE_CHUNK_SEQUENCES if chunk is None else chunk
    n = scaled.shape[0]
    if n < seq_len:
        raise ValueError(f"Only {n} usable rows; need at least {seq_len} for one sequence.")
    starts = np.arange(0, n - seq_len + 1, stride)
    offsets = np.arange(seq_len)
    errors = np.empty(len(starts), dtype=float)
    for i in range(0, len(starts), chunk):
        s = starts[i:i + chunk]
        seqs = scaled[s[:, None] + offsets[None, :]]
        recon = model.predict(seqs, batch_size=batch_size, verbose=0)
        errors[i:i + len(s)] = np.mean((seqs - recon) ** 2, axis=(1, 2))
    return errors


def load_threshold(base_dir) -> float:
    path = Path(base_dir) / 'outputs' / 'anomaly_detection_report.json'
    if not path.exists():
        raise FileNotFoundError(f"Threshold not found: {path}. Run detect_anomalies.py after training.")
    with open(path, 'r', encoding='utf-8-sig') as f:
        return float(json.load(f)['threshold_calculation']['threshold_value'])


def run_inference(input_csv: str, base_dir=ROOT, output_prefix: str = 'inference',
                  make_handoff: bool = True):
    """Score a telemetry CSV with the trained artifacts. Returns a summary
    dict, or None on failure (the reason is printed)."""
    import keras

    base = Path(base_dir).resolve()
    print('=' * 80)
    print(' ' * 25 + 'INFERENCE MODE')
    print('=' * 80)
    print('\nUsing existing trained artifacts (nothing refitted / retrained / recalculated):')
    print(f'  - Scaler:    {base / "models" / "feature_scaler.pkl"}')
    print(f'  - Filters:   {base / "models" / pc.FILTER_PARAMS_FILENAME}')
    print(f'  - Model:     {base / "models" / "lstm_autoencoder.keras"}')
    print(f'  - Threshold: {base / "outputs" / "anomaly_detection_report.json"}\n')

    try:
        meta = sg.verify_schema(base)
        feature_names = meta['feature_names']
        seq_len, stride = int(meta['sequence_length']), int(meta['stride'])
        warmup = int(meta['warmup_rows_removed'])

        work = base / pc.INFERENCE_WORK_DIR
        proc_dir, feat_dir = work / 'processed', work / 'features'

        print('[1/5] Preprocessing (saved filter parameters)...')
        pre = TelemetryPreprocessor(str(input_csv), filter_params_path=str(base / 'models' / pc.FILTER_PARAMS_FILENAME),
                                    fit_filters=False)
        if not pre.run(str(proc_dir)):
            print('X Preprocessing failed')
            return None
        sg.check_sampling_rate(pre.stats['sampling_rate_hz'], meta)
        cleaned_csv = proc_dir / 'cleaned_telemetry.csv'

        print('\n[2/5] Feature extraction...')
        if not run_phase2(str(cleaned_csv), str(feat_dir)):
            print('X Feature extraction failed validation')
            return None
        features_csv = feat_dir / 'extracted_features.csv'
        df_features = pd.read_csv(features_csv).iloc[warmup:].reset_index(drop=True)
        if df_features[feature_names].isna().any().any():
            raise ValueError('NaNs remain in features after removing the warm-up rows.')

        print('\n[3/5] Scaling + sequences (saved scaler)...')
        with open(base / 'models' / 'feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        if scaler.n_features_in_ != len(feature_names):
            raise ValueError(f'Scaler expects {scaler.n_features_in_} features, model metadata lists '
                             f'{len(feature_names)}.')
        scaled = scaler.transform(df_features[feature_names].values)   # by NAME and trained ORDER

        print('[4/5] Loading model and scoring...')
        model = keras.models.load_model(base / 'models' / 'lstm_autoencoder.keras')
        errors = score_scaled(model, scaled, seq_len, stride)
        print(f'  ✓ {len(errors)} sequences scored (length {seq_len}, stride {stride})')

        print('[5/5] Loading threshold and classifying...')
        threshold = load_threshold(base)
        classifications = np.where(errors <= threshold, 'Normal', 'Anomaly')
        n_norm = int(np.sum(classifications == 'Normal'))
        n_anom = int(np.sum(classifications == 'Anomaly'))
        print(f'  ✓ Threshold loaded: {threshold:.6f}')

        out_dir = base / 'outputs'
        out_dir.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame({'sequence_index': np.arange(len(errors)),
                                   'reconstruction_error': errors,
                                   'threshold': threshold,
                                   'classification': classifications})
        results_path = out_dir / f'{output_prefix}_results.csv'
        results_df.to_csv(results_path, index=False)

        if make_handoff:
            handoff_name = ('downstream_health_vector.csv' if output_prefix == 'inference'
                            else f'{output_prefix}_downstream_health_vector.csv')
            create_downstream_handoff(str(results_path), str(cleaned_csv), str(features_csv),
                                      str(out_dir / handoff_name), meta=meta)

        with open(proc_dir / 'preprocessing_report.json', 'r') as f:
            noise = json.load(f).get('noise_estimate', {})
        with open(out_dir / f'{output_prefix}_noise_estimate.json', 'w') as f:
            json.dump(noise, f, indent=2)

        summary = {
            'input_file': str(input_csv), 'total_sequences': int(len(errors)),
            'normal_count': n_norm, 'anomaly_count': n_anom, 'threshold_used': threshold,
            'model_retrained': False, 'scaler_refit': False, 'threshold_recalculated': False,
            'filters_rederived': False,
        }
        with open(out_dir / f'{output_prefix}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print('\n' + '=' * 80)
        print(' ' * 25 + 'INFERENCE COMPLETE')
        print('=' * 80)
        print(f'\nResults:\n  Total sequences: {len(errors)}')
        print(f'  Normal:  {n_norm} ({n_norm / len(errors) * 100:.2f}%)')
        print(f'  Anomaly: {n_anom} ({n_anom / len(errors) * 100:.2f}%)')
        print(f'\nOutput: {results_path}\n')
        return summary

    except (sg.SchemaMismatchError, FileNotFoundError, ValueError) as e:
        print(f'\nX Inference stopped: {e}')
        return None
    except Exception as e:
        print(f'\nX Inference failed: {e}')
        import traceback
        traceback.print_exc()
        return None
