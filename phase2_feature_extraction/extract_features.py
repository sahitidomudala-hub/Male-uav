"""
Phase 2: Feature Extraction for Engine Telemetry

Extracts time-domain features from 50 Hz engine telemetry data.
Designed for slow telemetry signals (NOT high-frequency vibration).

Features:
- Current/raw values
- Moving statistics (trailing windows)
- Rate of change (backward difference)
- Deviation metrics

All features are CAUSAL (no future-looking calculations).
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List

# channel_config.py lives at the repo root, one level up from this file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from channel_config import (get_moving_stat_channels, get_rate_channels, get_raw_only_channels,
                            get_units_dict, CHANNEL_CONFIG, TIME_COLUMN)
import pipeline_config as _pc

_SFX = _pc.get_feature_suffixes()   # {'raw': '_raw', 'ma': '_ma', 'std': '_std', 'rate': '_rate'}


class TelemetryFeatureExtractor:
    """
    Extract features from 50 Hz engine telemetry.
    
    Designed for slow-varying signals like torque, RPM, temperatures, pressures.
    NOT for high-frequency vibration analysis.
    """
    
    def __init__(self, window_size: int = None):
        """
        Initialize feature extractor.
        
        Args:
            window_size: Number of samples for moving windows. Default None:
                derived from the input file as MOVING_WINDOW_SECONDS x the
                sampling rate detected in its Time column (pipeline_config).
        """
        self.window_size = window_size
        self.sampling_rate_hz = None
        self.feature_definitions = []
        
        # Channel groupings now come from channel_config.py (single source of
        # truth shared with Phase 1's filtering step and the downstream
        # handoff). To change which channels get which feature types, edit
        # channel_config.py only -- do not hardcode lists here.
        self.moving_feature_channels = get_moving_stat_channels()
        self.rate_feature_channels = get_rate_channels()
        self.raw_only_channels = get_raw_only_channels()
        
        print(f"Telemetry Feature Extractor initialized:")
        print(f"  Moving window: {window_size if window_size else str(_pc.MOVING_WINDOW_SECONDS) + ' s (samples derived from the file sampling rate)'}")
        print(f"  Moving features for: {', '.join(self.moving_feature_channels)}")
        print(f"  Rate features for: {', '.join(self.rate_feature_channels)}")
        print(f"  Raw only: {', '.join(self.raw_only_channels)}")
    
    def extract_features(self, cleaned_telemetry_path: str) -> pd.DataFrame:
        """
        Extract features from cleaned telemetry.
        
        Args:
            cleaned_telemetry_path: Path to Phase 1 cleaned CSV
        
        Returns:
            DataFrame with extracted features
        """
        print(f"\nLoading cleaned telemetry: {cleaned_telemetry_path}")
        df = pd.read_csv(cleaned_telemetry_path)
        
        print(f"  Loaded: {len(df)} rows × {len(df.columns)} columns")
        print(f"  Duration: {df['Time'].max():.2f} seconds")

        # Window length in samples: from the file's own sampling rate unless
        # the caller fixed it explicitly.
        dt = df[TIME_COLUMN].diff().iloc[1:].median()
        self.sampling_rate_hz = float(1.0 / dt)
        if self.window_size is None:
            self.window_size = max(2, int(round(_pc.MOVING_WINDOW_SECONDS * self.sampling_rate_hz)))
        print(f"  Sampling rate {self.sampling_rate_hz:.2f} Hz -> moving window {self.window_size} samples "
              f"({self.window_size / self.sampling_rate_hz:.2f} s)")
        
        # Initialize feature dataframe with Time
        features = pd.DataFrame()
        features['Time'] = df['Time']
        
        self._add_definition('Time', ['Time'], 'Direct copy', None, 'Timestamp', True,
                            'Preserve temporal alignment')
        
        print(f"\nExtracting features...")
        
        # ==============================
        # Moving Statistics Features
        # ==============================
        for channel in self.moving_feature_channels:
            if channel not in df.columns:
                print(f"  ⚠ Skipping {channel} (not in dataset)")
                continue
            
            # Raw value
            features[f'{channel}{_SFX["raw"]}'] = df[channel]
            self._add_definition(
                f'{channel}{_SFX["raw"]}', [channel], 'Direct value', None,
                self._get_units(channel), True,
                f'Current {channel} measurement'
            )
            
            # Moving average (trailing)
            features[f'{channel}{_SFX["ma"]}'] = df[channel].rolling(
                window=self.window_size,
                min_periods=_pc.ROLLING_MEAN_MIN_PERIODS  # Allow calculation from first sample
            ).mean()
            self._add_definition(
                f'{channel}{_SFX["ma"]}', [channel], 
                f'Trailing mean over {self.window_size} samples',
                self.window_size, self._get_units(channel), True,
                f'Recent average {channel} (smoothed)'
            )
            
            # Moving std deviation (trailing)
            features[f'{channel}{_SFX["std"]}'] = df[channel].rolling(
                window=self.window_size,
                min_periods=_pc.ROLLING_STD_MIN_PERIODS  # Need at least 2 for std
            ).std()
            self._add_definition(
                f'{channel}{_SFX["std"]}', [channel],
                f'Trailing std deviation over {self.window_size} samples',
                self.window_size, self._get_units(channel), True,
                f'{channel} variability/stability indicator'
            )
        
        # ==============================
        # Rate of Change Features
        # ==============================
        for channel in self.rate_feature_channels:
            if channel not in df.columns:
                print(f"  ⚠ Skipping {channel}_rate (not in dataset)")
                continue
            
            # Only add raw if not already added
            if f'{channel}{_SFX["raw"]}' not in features.columns:
                features[f'{channel}{_SFX["raw"]}'] = df[channel]
                self._add_definition(
                    f'{channel}{_SFX["raw"]}', [channel], 'Direct value', None,
                    self._get_units(channel), True,
                    f'Current {channel} measurement'
                )
            
            # Rate of change (backward difference)
            features[f'{channel}{_SFX["rate"]}'] = df[channel].diff()
            self._add_definition(
                f'{channel}{_SFX["rate"]}', [channel],
                'Backward difference: value[t] - value[t-1]',
                1, f'{self._get_units(channel)}/sample', True,
                f'{channel} rate of change (transient detection)'
            )
        
        # ==============================
        # Raw-Only Features
        # ==============================
        for channel in self.raw_only_channels:
            if channel not in df.columns:
                print(f"  ⚠ Skipping {channel} (not in dataset)")
                continue
            
            features[f'{channel}{_SFX["raw"]}'] = df[channel]
            self._add_definition(
                f'{channel}{_SFX["raw"]}', [channel], 'Direct value', None,
                self._get_units(channel), True,
                f'Current {channel} measurement'
            )
        
        expected = _pc.compute_expected_feature_count(CHANNEL_CONFIG) + 1  # + Time
        self.expected_feature_count = expected
        print(f"  ✓ Extracted {len(features.columns)} feature columns incl. Time "
              f"(expected from channel_config: {expected})")
        
        return features
    
    def _add_definition(self, name: str, source: List[str], calculation: str,
                       window: int, units: str, causal: bool, justification: str):
        """Record feature definition for documentation."""
        self.feature_definitions.append({
            'feature_name': name,
            'source_columns': source,
            'calculation': calculation,
            'window_size': window,
            'units': units,
            'causal': causal,
            'justification': justification
        })
    
    def _get_units(self, channel: str) -> str:
        """Return the unit string for a channel (single source: channel_config)."""
        if channel == TIME_COLUMN:
            return 's'
        return get_units_dict().get(channel, 'unknown')
    
    def validate_features(self, features: pd.DataFrame, 
                         original_telemetry_path: str) -> Dict:
        """
        Validate extracted features.
        
        Args:
            features: Feature DataFrame
            original_telemetry_path: Path to original cleaned telemetry
        
        Returns:
            Validation results dictionary
        """
        print("\nValidating features...")
        
        original = pd.read_csv(original_telemetry_path)
        
        validation = {
            'input_rows': len(original),
            'output_rows': len(features),
            'rows_match': len(original) == len(features),
            'input_columns': len(original.columns),
            'output_features': len(features.columns),
            'time_aligned': False,
            'time_ordered': False,
            'nan_counts': {},
            'inf_counts': {},
            'all_numeric': True,
            'no_unexpected_nans': True,
            'expected_feature_count': getattr(self, 'expected_feature_count', None),
            'feature_count_match': True
        }
        if validation['expected_feature_count'] is not None:
            validation['feature_count_match'] = (len(features.columns) == validation['expected_feature_count'])
            if validation['feature_count_match']:
                print(f"  ✓ Feature count matches channel_config: {len(features.columns)} (incl. Time)")
            else:
                print(f"  ✗ Feature count {len(features.columns)} != expected {validation['expected_feature_count']}")
        
        # Check row count
        if validation['rows_match']:
            print(f"  ✓ Row count preserved: {len(features)} rows")
        else:
            print(f"  ✗ Row count mismatch: {len(original)} → {len(features)}")
        
        # Check Time alignment
        if np.allclose(original['Time'].values, features['Time'].values):
            validation['time_aligned'] = True
            print(f"  ✓ Time column aligned with input")
        else:
            print(f"  ✗ Time column NOT aligned")
        
        # Check Time ordering
        time_diffs = features['Time'].diff()[1:]
        if (time_diffs > 0).all():
            validation['time_ordered'] = True
            print(f"  ✓ Time is monotonically increasing")
        else:
            print(f"  ✗ Time is NOT monotonic")
        
        # Check for NaN values
        for col in features.columns:
            nan_count = features[col].isna().sum()
            if nan_count > 0:
                validation['nan_counts'][col] = int(nan_count)
                # Expected NaNs in first samples of moving features
                if col.endswith(_SFX['ma']) or col.endswith(_SFX['std']):
                    if nan_count <= self.window_size:
                        print(f"  ✓ {col}: {nan_count} NaNs (expected warm-up)")
                    else:
                        print(f"  ⚠ {col}: {nan_count} NaNs (more than expected)")
                        validation['no_unexpected_nans'] = False
                elif col.endswith(_SFX['rate']):
                    if nan_count == 1:
                        print(f"  ✓ {col}: {nan_count} NaN (expected at t=0)")
                    else:
                        print(f"  ⚠ {col}: {nan_count} NaNs (unexpected)")
                        validation['no_unexpected_nans'] = False
                else:
                    print(f"  ⚠ {col}: {nan_count} NaNs (unexpected)")
                    validation['no_unexpected_nans'] = False
        
        if not validation['nan_counts']:
            print(f"  ✓ No NaN values (unusual but acceptable)")
        
        # Check for Inf values
        for col in features.select_dtypes(include=[np.number]).columns:
            inf_count = np.isinf(features[col]).sum()
            if inf_count > 0:
                validation['inf_counts'][col] = int(inf_count)
                print(f"  ✗ {col}: {inf_count} Inf values")
                validation['all_numeric'] = False
        
        if not validation['inf_counts']:
            print(f"  ✓ No Inf values")
        
        # Overall pass/fail
        validation['passed'] = (
            validation['rows_match'] and
            validation['time_aligned'] and
            validation['time_ordered'] and
            validation['all_numeric'] and
            validation['no_unexpected_nans'] and
            validation['feature_count_match']
        )
        
        return validation
    
    def save_feature_definitions(self, output_path: str):
        """Save feature definitions to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.feature_definitions, f, indent=2)
        print(f"  ✓ Feature definitions saved: {output_path}")


def run_phase2(cleaned_telemetry_path: str, output_dir: str = 'data/features') -> bool:
    """
    Run Phase 2 feature extraction.
    
    Args:
        cleaned_telemetry_path: Path to Phase 1 cleaned telemetry
        output_dir: Output directory for features
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print("="*80)
        print(" "*25 + "PHASE 2 - FEATURE EXTRACTION")
        print("="*80)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize extractor
        extractor = TelemetryFeatureExtractor()
        
        # Extract features
        features = extractor.extract_features(cleaned_telemetry_path)
        
        # Save features
        features_csv = output_path / 'extracted_features.csv'
        features.to_csv(features_csv, index=False)
        print(f"\n✓ Features saved: {features_csv}")
        print(f"  Shape: {features.shape[0]} rows × {features.shape[1]} columns")
        
        # Save feature definitions
        definitions_json = Path(__file__).resolve().parent / 'feature_definitions.json'
        extractor.save_feature_definitions(str(definitions_json))
        
        # Validate
        validation = extractor.validate_features(features, cleaned_telemetry_path)
        
        # Save validation results
        validation_json = output_path / 'phase2_validation.json'
        with open(validation_json, 'w') as f:
            json.dump(validation, f, indent=2)
        print(f"  ✓ Validation results saved: {validation_json}")
        
        print("\n" + "="*80)
        print(" "*28 + "PHASE 2 COMPLETE")
        print("="*80)
        print(f"\nInput:  {len(features)} rows")
        print(f"Output: {len(features)} rows × {len(features.columns)} features")
        print(f"Validation: {'PASSED' if validation['passed'] else 'FAILED'}")
        
        return validation['passed']
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(" "*28 + "PHASE 2 FAILED")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for Phase 2."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_features.py <cleaned_telemetry_csv>")
        sys.exit(1)
    
    cleaned_csv = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/features'
    
    success = run_phase2(cleaned_csv, output_dir)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
