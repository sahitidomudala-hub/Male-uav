"""
Phase 5: Anomaly Detection Validation
======================================

Validates Phase 5 implementation and outputs.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import schema_guard as _sg
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys


def validate_phase5(base_dir: str) -> bool:
    """
    Validate Phase 5 anomaly detection implementation.
    
    Args:
        base_dir: Base directory (integrated_pipeline/)
        
    Returns:
        True if all validations pass
    """
    base_dir = Path(base_dir)
    # Config must match the trained artifacts (schema guard)
    try:
        _sg.verify_schema(base_dir)
        print("\n[0] Schema guard...\n  ✓ PASS: configuration matches the trained artifacts")
    except _sg.SchemaMismatchError as _e:
        print(f"\n[0] Schema guard...\n  ✗ FAIL: {_e}")
        return False
    all_passed = True
    
    print("="*70)
    print("PHASE 5 VALIDATION - ANOMALY DETECTION")
    print("="*70)
    
    # Check 1: Results file exists
    print("\n[1/10] Checking results file exists...")
    results_path = base_dir / 'outputs' / 'anomaly_detection_results.csv'
    if not results_path.exists():
        print(f"  ✗ FAIL: Results not found at {results_path}")
        all_passed = False
    else:
        print(f"  ✓ PASS: Results file exists")
    
    # Check 2: Report file exists
    print("\n[2/10] Checking report file exists...")
    report_path = base_dir / 'outputs' / 'anomaly_detection_report.json'
    if not report_path.exists():
        print(f"  ✗ FAIL: Report not found at {report_path}")
        all_passed = False
    else:
        print(f"  ✓ PASS: Report file exists")
    
    # Load files
    results_df = pd.read_csv(results_path)
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    val_errors_df = pd.read_csv(base_dir / 'outputs' / 'reconstruction_errors_val.csv')
    test_errors_df = pd.read_csv(base_dir / 'outputs' / 'reconstruction_errors_test.csv')
    
    # Check 3: Required columns exist
    print("\n[3/10] Checking required columns...")
    required_cols = ['sequence_index', 'reconstruction_error', 'threshold', 'classification']
    missing = [col for col in required_cols if col not in results_df.columns]
    if missing:
        print(f"  ✗ FAIL: Missing columns: {missing}")
        all_passed = False
    else:
        print(f"  ✓ PASS: All required columns present")
    
    # Check 4: Every test sequence classified
    print("\n[4/10] Checking all test sequences classified...")
    n_test = len(test_errors_df)
    n_results = len(results_df)
    if n_results != n_test:
        print(f"  ✗ FAIL: Results count {n_results} != test count {n_test}")
        all_passed = False
    else:
        print(f"  ✓ PASS: All {n_test} test sequences classified")
    
    # Check 5: No NaN/Inf
    print("\n[5/10] Checking for NaN/Inf...")
    has_nan = results_df.isnull().any().any()
    has_inf = np.isinf(results_df.select_dtypes(include=[np.number])).any().any()
    if has_nan:
        print(f"  ✗ FAIL: Results contain NaN")
        all_passed = False
    elif has_inf:
        print(f"  ✗ FAIL: Results contain Inf")
        all_passed = False
    else:
        print(f"  ✓ PASS: No NaN/Inf in results")
    
    # Check 6: Threshold calculated from validation only
    print("\n[6/10] Checking threshold source...")
    if not report['threshold_calculation']['calculated_from'] == 'validation_errors_only':
        print(f"  ✗ FAIL: Threshold not calculated from validation only")
        all_passed = False
    elif report['data_integrity']['test_data_used_for_threshold']:
        print(f"  ✗ FAIL: Test data was used for threshold")
        all_passed = False
    else:
        print(f"  ✓ PASS: Threshold calculated from validation data only")
        print(f"    Method: {report['threshold_calculation']['method']}")
    
    # Check 7: Verify threshold value
    print("\n[7/10] Verifying threshold calculation...")
    val_errors = val_errors_df['reconstruction_error_mse'].values
    percentile = report['threshold_calculation']['percentile']
    reported_threshold = report['threshold_calculation']['threshold_value']
    calculated_threshold = np.percentile(val_errors, percentile)
    
    if not np.isclose(reported_threshold, calculated_threshold, rtol=1e-5):
        print(f"  ✗ FAIL: Threshold mismatch")
        print(f"    Reported: {reported_threshold:.6f}")
        print(f"    Calculated: {calculated_threshold:.6f}")
        all_passed = False
    else:
        print(f"  ✓ PASS: Threshold correctly calculated")
        print(f"    Threshold: {reported_threshold:.6f}")
        print(f"    Percentile: {percentile}")
    
    # Check 8: Verify classifications
    print("\n[8/10] Verifying classifications...")
    threshold = report['threshold_calculation']['threshold_value']
    test_errors = results_df['reconstruction_error'].values
    expected_classes = np.where(test_errors <= threshold, 'Normal', 'Anomaly')
    actual_classes = results_df['classification'].values
    
    if not np.array_equal(expected_classes, actual_classes):
        print(f"  ✗ FAIL: Classifications don't match expected")
        all_passed = False
    else:
        print(f"  ✓ PASS: All classifications correct")
        normal_count = np.sum(actual_classes == 'Normal')
        anomaly_count = np.sum(actual_classes == 'Anomaly')
        print(f"    Normal: {normal_count}")
        print(f"    Anomaly: {anomaly_count}")
    
    # Check 9: No model retraining
    print("\n[9/10] Checking no model retraining...")
    if report['data_integrity']['model_retrained']:
        print(f"  ✗ FAIL: Model was retrained")
        all_passed = False
    else:
        print(f"  ✓ PASS: Model NOT retrained")
    
    # Check 10: No synthetic faults
    print("\n[10/10] Checking no synthetic faults...")
    if report['data_integrity']['synthetic_faults_generated']:
        print(f"  ✗ FAIL: Synthetic faults were generated")
        all_passed = False
    else:
        print(f"  ✓ PASS: No synthetic faults generated")
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("VALIDATION RESULT: ALL CHECKS PASSED ✓")
    else:
        print("VALIDATION RESULT: SOME CHECKS FAILED ✗")
    print("="*70)
    
    return all_passed


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Phase 5')
    parser.add_argument('--base-dir', type=str,
                        default='.',
                        help='Base directory')
    
    args = parser.parse_args()
    
    success = validate_phase5(args.base_dir)
    sys.exit(0 if success else 1)
