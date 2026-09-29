"""
Phase 5: Anomaly Detection via Reconstruction Error Thresholding
=================================================================

Uses reconstruction errors from Phase 4 LSTM autoencoder to classify
sequences as Normal or Anomaly based on a validation-derived threshold.

Methodology:
    1. Calculate threshold from VALIDATION errors only
    2. Apply threshold to TEST errors
    3. Classify: error <= threshold → Normal, error > threshold → Anomaly

NO ground-truth labels are used.
NO synthetic faults are generated.
NO performance metrics (accuracy, precision, recall) are calculated.
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import pipeline_config as _pc
import schema_guard as _sg
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple


def calculate_threshold_percentile(validation_errors: np.ndarray,
                                   percentile: float = None) -> Tuple[float, Dict]:
    """
    Calculate anomaly threshold using percentile method on validation errors.
    
    Method: Threshold = percentile(validation_errors, p)
    
    Rationale:
    - Model trained on normal data
    - Validation set should also be mostly normal
    - Set threshold to capture most validation sequences as normal
    - Sequences beyond this percentile are classified as anomalies
    
    Args:
        validation_errors: Reconstruction errors from validation set
        percentile: Percentile value (default: 95.0)
        
    Returns:
        threshold: Threshold value
        stats: Statistics dictionary
    """
    percentile = _pc.THRESHOLD_PERCENTILE if percentile is None else percentile
    threshold = np.percentile(validation_errors, percentile)
    
    stats = {
        'method': f'{percentile}th percentile of validation errors',
        'percentile': percentile,
        'threshold': float(threshold),
        'validation_statistics': {
            'count': int(len(validation_errors)),
            'min': float(validation_errors.min()),
            'max': float(validation_errors.max()),
            'mean': float(validation_errors.mean()),
            'median': float(np.median(validation_errors)),
            'std': float(validation_errors.std()),
            'percentile_25': float(np.percentile(validation_errors, 25)),
            'percentile_50': float(np.percentile(validation_errors, 50)),
            'percentile_75': float(np.percentile(validation_errors, 75)),
            'percentile_90': float(np.percentile(validation_errors, 90)),
            'percentile_95': float(np.percentile(validation_errors, 95)),
            'percentile_99': float(np.percentile(validation_errors, 99))
        }
    }
    
    return threshold, stats


def classify_sequences(test_errors: np.ndarray,
                      threshold: float) -> Tuple[np.ndarray, Dict]:
    """
    Classify test sequences as Normal or Anomaly based on threshold.
    
    Args:
        test_errors: Reconstruction errors from test set
        threshold: Threshold value (from validation set)
        
    Returns:
        classifications: Array of 'Normal' or 'Anomaly' strings
        stats: Classification statistics
    """
    classifications = np.where(test_errors <= threshold, 'Normal', 'Anomaly')
    
    n_total = len(test_errors)
    n_normal = np.sum(classifications == 'Normal')
    n_anomaly = np.sum(classifications == 'Anomaly')
    pct_anomaly = (n_anomaly / n_total) * 100
    
    stats = {
        'total_sequences': int(n_total),
        'normal_count': int(n_normal),
        'anomaly_count': int(n_anomaly),
        'normal_percentage': float((n_normal / n_total) * 100),
        'anomaly_percentage': float(pct_anomaly),
        'test_error_statistics': {
            'min': float(test_errors.min()),
            'max': float(test_errors.max()),
            'mean': float(test_errors.mean()),
            'median': float(np.median(test_errors)),
            'std': float(test_errors.std())
        }
    }
    
    return classifications, stats


def run_anomaly_detection(base_dir: str,
                          percentile: float = None) -> Dict:
    """
    Main Phase 5 anomaly detection pipeline.
    
    Args:
        base_dir: Base directory (integrated_pipeline/)
        percentile: Percentile for threshold calculation
        
    Returns:
        Results dictionary
    """
    percentile = _pc.THRESHOLD_PERCENTILE if percentile is None else percentile
    base_dir = Path(base_dir)
    _sg.verify_schema(base_dir)  # config must match the trained artifacts
    
    print("="*70)
    print("PHASE 5: ANOMALY DETECTION")
    print("="*70)
    
    # Load reconstruction errors
    print("\n[1/5] Loading reconstruction errors...")
    val_errors_path = base_dir / 'outputs' / 'reconstruction_errors_val.csv'
    test_errors_path = base_dir / 'outputs' / 'reconstruction_errors_test.csv'
    
    val_df = pd.read_csv(val_errors_path)
    test_df = pd.read_csv(test_errors_path)
    
    val_errors = val_df['reconstruction_error_mse'].values
    test_errors = test_df['reconstruction_error_mse'].values
    
    print(f"  Validation errors: {len(val_errors)}")
    print(f"  Test errors: {len(test_errors)}")
    
    # Verify no NaN/Inf
    assert not np.any(np.isnan(val_errors)), "Validation errors contain NaN"
    assert not np.any(np.isinf(val_errors)), "Validation errors contain Inf"
    assert not np.any(np.isnan(test_errors)), "Test errors contain NaN"
    assert not np.any(np.isinf(test_errors)), "Test errors contain Inf"
    print("  ✓ No NaN/Inf in errors")
    
    # Calculate threshold from validation errors ONLY
    print(f"\n[2/5] Calculating threshold ({percentile}th percentile)...")
    threshold, threshold_stats = calculate_threshold_percentile(val_errors, percentile)
    
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Validation error range: [{val_errors.min():.6f}, {val_errors.max():.6f}]")
    print(f"  Validation error mean: {val_errors.mean():.6f}")
    print(f"  ✓ Threshold calculated from validation data ONLY")
    
    # Classify test sequences
    print("\n[3/5] Classifying test sequences...")
    classifications, class_stats = classify_sequences(test_errors, threshold)
    
    print(f"  Total test sequences: {class_stats['total_sequences']}")
    print(f"  Normal: {class_stats['normal_count']} ({class_stats['normal_percentage']:.2f}%)")
    print(f"  Anomaly: {class_stats['anomaly_count']} ({class_stats['anomaly_percentage']:.2f}%)")
    
    # Create results DataFrame
    print("\n[4/5] Generating results...")
    results_df = pd.DataFrame({
        'sequence_index': test_df['sequence_index'],
        'reconstruction_error': test_errors,
        'threshold': threshold,
        'classification': classifications
    })
    
    # Save results
    results_path = base_dir / 'outputs' / 'anomaly_detection_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"  Results saved: {results_path}")
    
    # Create report
    print("\n[5/5] Generating report...")
    report = {
        'phase': 'Phase 5: Anomaly Detection',
        'method': 'Reconstruction Error Thresholding',
        'threshold_calculation': {
            'method': threshold_stats['method'],
            'percentile': percentile,
            'threshold_value': threshold,
            'calculated_from': 'validation_errors_only',
            'validation_statistics': threshold_stats['validation_statistics']
        },
        'classification_results': {
            'total_test_sequences': class_stats['total_sequences'],
            'normal_count': class_stats['normal_count'],
            'anomaly_count': class_stats['anomaly_count'],
            'normal_percentage': class_stats['normal_percentage'],
            'anomaly_percentage': class_stats['anomaly_percentage'],
            'test_error_statistics': class_stats['test_error_statistics']
        },
        'data_integrity': {
            'test_data_used_for_threshold': False,
            'synthetic_faults_generated': False,
            'ground_truth_labels_available': False,
            'model_retrained': False
        },
        'interpretation': {
            'note': 'Classifications are based on reconstruction error thresholding, NOT ground-truth labels.',
            'anomaly_definition': 'Sequences with reconstruction error exceeding validation-derived threshold.',
            'normal_definition': 'Sequences with reconstruction error within expected range based on validation data.',
            'limitation': 'Without validated fault labels, detected anomalies require further investigation to confirm actual faults.'
        },
        'files': {
            'results': 'outputs/anomaly_detection_results.csv',
            'validation_errors': 'outputs/reconstruction_errors_val.csv',
            'test_errors': 'outputs/reconstruction_errors_test.csv',
            'model': 'models/lstm_autoencoder.keras'
        }
    }
    
    report_path = base_dir / 'outputs' / 'anomaly_detection_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")
    
    print("\n" + "="*70)
    print("PHASE 5 COMPLETE")
    print("="*70)
    print(f"Threshold: {threshold:.6f}")
    print(f"Normal sequences: {class_stats['normal_count']} ({class_stats['normal_percentage']:.2f}%)")
    print(f"Anomaly sequences: {class_stats['anomaly_count']} ({class_stats['anomaly_percentage']:.2f}%)")
    print("="*70)
    
    return report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 5: Anomaly Detection')
    parser.add_argument('--base-dir', type=str,
                        default='.',
                        help='Base directory (integrated_pipeline/)')
    parser.add_argument('--percentile', type=float, default=_pc.THRESHOLD_PERCENTILE,
                        help=f'Percentile for threshold calculation (default from pipeline_config: {_pc.THRESHOLD_PERCENTILE})')
    
    args = parser.parse_args()
    
    results = run_anomaly_detection(args.base_dir, args.percentile)
