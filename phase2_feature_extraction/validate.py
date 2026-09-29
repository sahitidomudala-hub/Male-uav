"""
Phase 2: Feature Extraction Validation

Validates Phase 2 feature extraction outputs.
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
from pathlib import Path
import argparse
import sys


def validate_phase2(cleaned_telemetry_path: str, features_dir: str = 'data/features') -> bool:
    """
    Run all Phase 2 validation checks.
    
    Args:
        cleaned_telemetry_path: Path to Phase 1 cleaned telemetry
        features_dir: Directory containing feature outputs
    
    Returns:
        True if all validations pass, False otherwise
    """
    print("="*80)
    print(" "*23 + "PHASE 2 - VALIDATION")
    print("="*80)
    
    cleaned_path = Path(cleaned_telemetry_path)
    features_path = Path(features_dir)
    
    if not cleaned_path.exists():
        print(f"ERROR: Cleaned telemetry not found: {cleaned_path}")
        return False
    
    print(f"\nInput (cleaned telemetry): {cleaned_path}")
    print(f"Output directory: {features_path}")
    
    features_csv = features_path / 'extracted_features.csv'
    definitions_json = Path('phase2_feature_extraction') / 'feature_definitions.json'
    validation_json = features_path / 'phase2_validation.json'
    
    all_passed = True
    
    # [1/10] Check output files exist
    print("\n[1/10] Checking output files exist...")
    if not features_csv.exists():
        print(f"  ✗ FAIL: {features_csv} not found")
        all_passed = False
    else:
        print(f"  ✓ PASS: {features_csv} exists")
    
    if not definitions_json.exists():
        print(f"  ✗ FAIL: {definitions_json} not found")
        all_passed = False
    else:
        print(f"  ✓ PASS: {definitions_json} exists")
    
    if not all_passed:
        print("\nValidation aborted: Missing output files")
        return False
    
    # Load data
    print("\n[2/10] Loading data...")
    cleaned = pd.read_csv(cleaned_path)
    features = pd.read_csv(features_csv)
    
    with open(definitions_json, 'r') as f:
        definitions = json.load(f)
    
    print(f"  Cleaned telemetry: {len(cleaned)} rows × {len(cleaned.columns)} columns")
    print(f"  Features: {len(features)} rows × {len(features.columns)} columns")
    print(f"  Feature definitions: {len(definitions)} features")

    expected_count = _pc.compute_expected_feature_count(_cc.CHANNEL_CONFIG)
    if len(features.columns) - 1 == expected_count:
        print(f"  ✓ PASS: {expected_count} model features (+ Time), as computed from channel_config")
    else:
        print(f"  ✗ FAIL: {len(features.columns) - 1} features found, channel_config implies {expected_count}")
        all_passed = False
    
    # [3/10] Row count maintained
    print("\n[3/10] Validating row count...")
    if len(cleaned) == len(features):
        print(f"  ✓ PASS: Row count maintained ({len(features)} rows)")
    else:
        print(f"  ✗ FAIL: Row count mismatch (input={len(cleaned)}, output={len(features)})")
        all_passed = False
    
    # [4/10] Time alignment
    print("\n[4/10] Validating Time alignment...")
    if 'Time' not in features.columns:
        print(f"  ✗ FAIL: Time column missing")
        all_passed = False
    elif np.allclose(cleaned['Time'].values, features['Time'].values):
        print(f"  ✓ PASS: Time aligned with input")
    else:
        print(f"  ✗ FAIL: Time NOT aligned")
        all_passed = False
    
    # [5/10] Time ordering
    print("\n[5/10] Validating Time ordering...")
    time_diffs = features['Time'].diff()[1:]
    if (time_diffs > 0).all():
        print(f"  ✓ PASS: Time is monotonically increasing")
    else:
        print(f"  ✗ FAIL: Time is NOT monotonic")
        all_passed = False
    
    # [6/10] No future-looking calculations
    print("\n[6/10] Checking for non-causal features...")
    non_causal = [d for d in definitions if not d.get('causal', True)]
    if non_causal:
        print(f"  ✗ FAIL: Found non-causal features:")
        for d in non_causal:
            print(f"    - {d['feature_name']}")
        all_passed = False
    else:
        print(f"  ✓ PASS: All features are causal (no future-looking)")
    
    # [7/10] Feature definitions match implementation
    print("\n[7/10] Validating feature definitions...")
    feature_cols = set(features.columns)
    defined_features = set(d['feature_name'] for d in definitions)
    
    if feature_cols == defined_features:
        print(f"  ✓ PASS: All {len(feature_cols)} features documented")
    else:
        missing = feature_cols - defined_features
        extra = defined_features - feature_cols
        if missing:
            print(f"  ✗ FAIL: Undocumented features: {missing}")
            all_passed = False
        if extra:
            print(f"  ✗ FAIL: Documented but not present: {extra}")
            all_passed = False
    
    # [8/10] No unavailable variables used
    print("\n[8/10] Checking for unavailable source columns...")
    available_cols = set(cleaned.columns)
    unavailable_used = []
    
    for d in definitions:
        sources = d.get('source_columns', [])
        for src in sources:
            if src not in available_cols and src != d['feature_name']:
                unavailable_used.append((d['feature_name'], src))
    
    if unavailable_used:
        print(f"  ✗ FAIL: Features using unavailable sources:")
        for feat, src in unavailable_used:
            print(f"    - {feat} uses {src}")
        all_passed = False
    else:
        print(f"  ✓ PASS: All features use available columns only")
    
    # [9/10] NaN/Inf validation
    print("\n[9/10] Validating numeric values...")
    
    # Expected NaNs
    expected_nans = {
        'std': max(_pc.ROLLING_STD_MIN_PERIODS - 1, 1),  # rolling-std start-up
        'rate': 1                                        # first backward difference
    }
    
    nan_ok = True
    inf_ok = True
    
    for col in features.columns:
        nan_count = features[col].isna().sum()
        inf_count = np.isinf(features[col]).sum() if features[col].dtype in [np.float64, np.float32] else 0
        
        # Check NaNs
        if nan_count > 0:
            expected = False
            for suffix, max_nans in expected_nans.items():
                if col.endswith(f'_{suffix}') and nan_count <= max_nans:
                    expected = True
                    break
            
            if not expected:
                print(f"  ⚠ {col}: {nan_count} unexpected NaNs")
                nan_ok = False
        
        # Check Infs
        if inf_count > 0:
            print(f"  ✗ {col}: {inf_count} Inf values")
            inf_ok = False
    
    if nan_ok:
        print(f"  ✓ PASS: NaN values only in expected warm-up periods")
    else:
        print(f"  ✗ FAIL: Unexpected NaN values found")
        all_passed = False
    
    if inf_ok:
        print(f"  ✓ PASS: No Inf values")
    else:
        print(f"  ✗ FAIL: Inf values found")
        all_passed = False
    
    # [10/10] Phase 1 output not modified
    print("\n[10/10] Verifying Phase 1 output integrity...")
    
    # Re-read to check it hasn't been modified
    cleaned_check = pd.read_csv(cleaned_path)
    if cleaned.equals(cleaned_check):
        print(f"  ✓ PASS: Phase 1 cleaned telemetry unchanged")
    else:
        print(f"  ✗ FAIL: Phase 1 output was modified!")
        all_passed = False
    
    # Final result
    print("\n" + "="*80)
    if all_passed:
        print(" "*25 + "ALL VALIDATIONS PASSED")
    else:
        print(" "*25 + "SOME VALIDATIONS FAILED")
    print("="*80)
    
    return all_passed


def main():
    """Main entry point for Phase 2 validation."""
    parser = argparse.ArgumentParser(
        description='Validate Phase 2 feature extraction outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python validate.py --input "data/processed/cleaned_telemetry.csv"
    python validate.py --input "cleaned.csv" --features-dir "results/features"
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Path to Phase 1 cleaned telemetry CSV'
    )
    
    parser.add_argument(
        '--features-dir',
        default='data/features',
        help='Directory containing feature outputs (default: data/features)'
    )
    
    args = parser.parse_args()
    
    success = validate_phase2(args.input, args.features_dir)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
