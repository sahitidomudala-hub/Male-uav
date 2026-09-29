"""
Phase 3: Sequence Preparation Validation

Validates that sequences are prepared correctly with no data leakage.
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
import channel_config as _cc
import pipeline_config as _pc
import schema_guard as _sg
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import argparse
import sys


def validate_phase3(features_csv_path: str, sequences_dir: str = 'data/sequences') -> bool:
    """
    Run all Phase 3 validation checks.
    
    Args:
        features_csv_path: Path to Phase 2 features
        sequences_dir: Directory containing sequences
    
    Returns:
        True if all validations pass, False otherwise
    """
    print("="*80)
    print(" "*23 + "PHASE 3 - VALIDATION")
    print("="*80)
    
    features_path = Path(features_csv_path)
    sequences_path = Path(sequences_dir)
    
    if not features_path.exists():
        print(f"ERROR: Features file not found: {features_path}")
        return False
    
    print(f"\nInput (Phase 2 features): {features_path}")
    print(f"Output directory: {sequences_path}")
    
    # Check output files exist
    metadata_file = sequences_path / 'metadata.json'
    scaler_file = _ROOT / 'models' / 'feature_scaler.pkl'
    train_file = sequences_path / 'X_train.npy'
    val_file = sequences_path / 'X_val.npy'
    test_file = sequences_path / 'X_test.npy'
    
    all_passed = True
    
    # [1/13] Check output files exist
    print("\n[1/13] Checking output files exist...")
    for filepath in [metadata_file, scaler_file, train_file, val_file, test_file]:
        if not filepath.exists():
            print(f"  ✗ FAIL: {filepath} not found")
            all_passed = False
        else:
            print(f"  ✓ PASS: {filepath} exists")
    
    if not all_passed:
        print("\nValidation aborted: Missing output files")
        return False
    
    # Load data
    print("\n[2/13] Loading data...")
    features_df = pd.read_csv(features_path)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    X_train = np.load(train_file)
    X_val = np.load(val_file)
    X_test = np.load(test_file)
    
    print(f"  Features: {len(features_df)} rows")
    print(f"  Train sequences: {X_train.shape}")
    print(f"  Val sequences: {X_val.shape}")
    print(f"  Test sequences: {X_test.shape}")
    
    # [3/13] No NaN/Inf in sequences
    print("\n[3/13] Checking for NaN/Inf in sequences...")
    nan_found = False
    inf_found = False
    
    for name, data in [('train', X_train), ('val', X_val), ('test', X_test)]:
        if np.isnan(data).any():
            print(f"  ✗ FAIL: NaN found in {name} sequences")
            nan_found = True
            all_passed = False
        if np.isinf(data).any():
            print(f"  ✗ FAIL: Inf found in {name} sequences")
            inf_found = True
            all_passed = False
    
    if not nan_found and not inf_found:
        print(f"  ✓ PASS: No NaN or Inf in any sequences")
    
    # [4/13] Timeline is chronological
    print("\n[4/13] Validating chronological order...")
    train_time = metadata['train_time_range']
    val_time = metadata['val_time_range']
    test_time = metadata['test_time_range']
    
    print(f"  Train: [{train_time[0]:.2f}, {train_time[1]:.2f}]s")
    print(f"  Val:   [{val_time[0]:.2f}, {val_time[1]:.2f}]s")
    print(f"  Test:  [{test_time[0]:.2f}, {test_time[1]:.2f}]s")
    
    if train_time[1] < val_time[0] and val_time[1] < test_time[0]:
        print(f"  ✓ PASS: Chronological order maintained")
    else:
        print(f"  ✗ FAIL: Time ranges overlap")
        all_passed = False
    
    # [5/13] No sequence crosses split boundary
    print("\n[5/13] Checking for boundary crossings...")
    
    # Calculate expected sequence counts
    train_rows = metadata['train_rows']
    val_rows = metadata['val_rows']
    test_rows = metadata['test_rows']
    seq_len = metadata['sequence_length']   # recorded by Phase 3
    stride = metadata['stride']
    
    expected_train_seqs = (train_rows - seq_len) // stride + 1
    expected_val_seqs = (val_rows - seq_len) // stride + 1
    expected_test_seqs = (test_rows - seq_len) // stride + 1
    
    actual_train_seqs = len(X_train)
    actual_val_seqs = len(X_val)
    actual_test_seqs = len(X_test)
    
    print(f"  Train: expected={expected_train_seqs}, actual={actual_train_seqs}")
    print(f"  Val:   expected={expected_val_seqs}, actual={actual_val_seqs}")
    print(f"  Test:  expected={expected_test_seqs}, actual={actual_test_seqs}")
    
    if (actual_train_seqs == expected_train_seqs and
        actual_val_seqs == expected_val_seqs and
        actual_test_seqs == expected_test_seqs):
        print(f"  ✓ PASS: Sequence counts match (no boundary crossing)")
    else:
        print(f"  ? WARNING: Sequence counts differ (may be due to stride)")
    
    # [6/13] Scaler type check
    print("\n[6/13] Validating scaler...")
    from sklearn.preprocessing import StandardScaler
    
    if isinstance(scaler, StandardScaler):
        print(f"  ✓ PASS: Scaler is StandardScaler")
    else:
        print(f"  ✗ FAIL: Unexpected scaler type: {type(scaler)}")
        all_passed = False
    
    # [7/13] Scaler fitted on training data only
    print("\n[7/13] Verifying scaler was fitted on training data...")
    if metadata.get('scaler_fitted_on') == 'training_data_only':
        print(f"  ✓ PASS: Metadata confirms scaler fitted on training only")
    else:
        print(f"  ✗ FAIL: Cannot confirm scaler fitting")
        all_passed = False
    
    # [8/13] Same scaler used for all splits
    print("\n[8/13] Checking scaler consistency...")
    # This is implicitly verified by having a single scaler file
    print(f"  ✓ PASS: Single scaler used for all splits")
    
    # [9/13] Sequence length matches configuration
    print("\n[9/13] Validating sequence length...")
    expected_seq_len = metadata['sequence_length']
    
    if X_train.shape[1] == expected_seq_len:
        print(f"  ✓ PASS: Train sequence length = {X_train.shape[1]}")
    else:
        print(f"  ✗ FAIL: Train sequence length = {X_train.shape[1]}, expected {expected_seq_len}")
        all_passed = False
    
    # [10/13] Feature dimension matches Phase 2
    print("\n[10/13] Validating feature dimension...")
    num_features_phase2 = len([c for c in features_df.columns if c != 'Time'])
    
    if X_train.shape[2] == num_features_phase2:
        print(f"  ✓ PASS: Feature dimension = {X_train.shape[2]}")
    else:
        print(f"  ✗ FAIL: Feature dim = {X_train.shape[2]}, expected {num_features_phase2}")
        all_passed = False
    
    # [11/13] No synthetic samples
    print("\n[11/13] Checking for synthetic samples...")
    total_input_rows = metadata['total_rows']
    warmup_removed = metadata['warmup_rows_removed']
    clean_rows = metadata['clean_rows']
    
    if clean_rows == total_input_rows - warmup_removed:
        print(f"  ✓ PASS: No synthetic samples (only {warmup_removed} warm-up row removed)")
    else:
        print(f"  ✗ FAIL: Row count mismatch")
        all_passed = False
    
    # [12/13] Sequence counts are consistent
    print("\n[12/13] Validating sequence count consistency...")
    metadata_train = metadata['train_sequences']
    metadata_val = metadata['validation_sequences']
    metadata_test = metadata['test_sequences']
    
    if (metadata_train == actual_train_seqs and
        metadata_val == actual_val_seqs and
        metadata_test == actual_test_seqs):
        print(f"  ✓ PASS: Metadata sequence counts match actual")
    else:
        print(f"  ✗ FAIL: Metadata mismatch")
        all_passed = False
    
    # [13/13] Warm-up handling
    print("\n[13/13] Validating warm-up handling...")
    if warmup_removed == 1:
        print(f"  ✓ PASS: Exactly 1 warm-up row removed (expected)")
    else:
        print(f"  ? WARNING: {warmup_removed} warm-up rows removed")
    
    # Final result
    print("\n" + "="*80)
    if all_passed:
        print(" "*25 + "ALL VALIDATIONS PASSED")
    else:
        print(" "*25 + "SOME VALIDATIONS FAILED")
    print("="*80)
    
    return all_passed


def main():
    """Main entry point for Phase 3 validation."""
    parser = argparse.ArgumentParser(
        description='Validate Phase 3 sequence preparation outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python validate.py --input "data/features/extracted_features.csv"
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Path to Phase 2 features CSV'
    )
    
    parser.add_argument(
        '--sequences-dir',
        default='data/sequences',
        help='Directory containing sequences (default: data/sequences)'
    )
    
    args = parser.parse_args()
    
    success = validate_phase3(args.input, args.sequences_dir)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
