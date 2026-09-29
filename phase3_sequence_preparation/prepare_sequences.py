"""
Phase 3: Sequence Preparation

Prepares time-series sequences for LSTM autoencoder training.

CRITICAL TEMPORAL SAFEGUARDS:
1. Chronological split (NO random shuffling)
2. Scaler fitted ONLY on training data
3. No sequences cross split boundaries
4. Temporal order preserved

Prevents:
- Data leakage (scaler sees future data)
- Temporal leakage (train sequences contain val/test data)
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Tuple, Dict
import sys
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import channel_config as _cc
import pipeline_config as _pc
import schema_guard as _sg


class SequencePreparator:
    """
    Prepare sequences for LSTM autoencoder with strict temporal safeguards.
    """
    
    def __init__(self, 
                 sequence_length: int = None,
                 stride: int = None,
                 train_ratio: float = None,
                 val_ratio: float = None,
                 test_ratio: float = None):
        """
        Initialize sequence preparator.
        
        Args:
            All parameters default to the values in pipeline_config.py
            (SEQUENCE_LENGTH, STRIDE, TRAIN/VAL/TEST_RATIO).
        """
        sequence_length = _pc.SEQUENCE_LENGTH if sequence_length is None else sequence_length
        stride = _pc.STRIDE if stride is None else stride
        train_ratio = _pc.TRAIN_RATIO if train_ratio is None else train_ratio
        val_ratio = _pc.VAL_RATIO if val_ratio is None else val_ratio
        test_ratio = _pc.TEST_RATIO if test_ratio is None else test_ratio
        self.sequence_length = sequence_length
        self.stride = stride
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        self.scaler = None
        self.feature_names = None
        self.metadata = {}
        
        print(f"Sequence Preparator initialized:")
        print(f"  Sequence length: {sequence_length} timesteps")
        print(f"  Stride: {stride}")
        print(f"  Split: {train_ratio*100:.0f}% train, {val_ratio*100:.0f}% val, {test_ratio*100:.0f}% test")
    
    def prepare_sequences(self, features_csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full sequence preparation pipeline.
        
        Args:
            features_csv_path: Path to Phase 2 features CSV
        
        Returns:
            (X_train, X_val, X_test): Normalized sequence arrays
        """
        print(f"\nLoading Phase 2 features: {features_csv_path}")
        df = pd.read_csv(features_csv_path)
        
        print(f"  Loaded: {len(df)} rows × {len(df.columns)} columns")
        print(f"  Time range: {df['Time'].min():.2f} to {df['Time'].max():.2f} seconds")
        
        # Store feature names (exclude Time)
        self.feature_names = [col for col in df.columns if col != 'Time']
        
        self.metadata['input_file'] = str(features_csv_path)
        self.metadata['total_rows'] = len(df)
        self.metadata['num_features'] = len(self.feature_names)
        self.metadata['feature_names'] = self.feature_names
        self.metadata['sequence_length'] = self.sequence_length
        self.metadata['stride'] = self.stride
        dt = df['Time'].diff().iloc[1:].median()
        self.metadata['sampling_rate_hz'] = float(1.0 / dt)
        self.metadata['model_input_channels'] = _cc.get_model_input_channels()
        self.metadata['passthrough_channels'] = _cc.get_passthrough_channels()
        self.metadata['expected_feature_count'] = _pc.compute_expected_feature_count(_cc.CHANNEL_CONFIG)
        self.metadata['config_fingerprint'] = _sg.config_fingerprint()
        if self.metadata['expected_feature_count'] != len(self.feature_names):
            raise ValueError(f"Feature count {len(self.feature_names)} != {self.metadata['expected_feature_count']} "
                             "expected from channel_config; Phase 2 output and config disagree.")
        
        # Handle warm-up NaNs
        df_clean = self._handle_warmup(df)
        
        # Chronological split
        train_df, val_df, test_df = self._chronological_split(df_clean)

        # Redundant (near-duplicate) channels, detected from TRAINING rows only.
        raw_sfx = _pc.FEATURE_SUFFIX_RAW
        raw_cols = {c[:-len(raw_sfx)]: c for c in self.feature_names if c.endswith(raw_sfx)}
        raw_view = train_df[list(raw_cols.values())].rename(columns={v: k for k, v in raw_cols.items()})
        groups = _sg.compute_redundant_groups(raw_view, list(raw_cols.keys()))
        self.metadata['redundant_groups'] = groups
        self.metadata['redundancy_cutoff'] = _pc.REDUNDANCY_CORRELATION_CUTOFF
        print(f"  Redundant channel groups (|r| >= {_pc.REDUNDANCY_CORRELATION_CUTOFF}, train only): {groups}")
        
        # Fit scaler on training data ONLY
        self._fit_scaler(train_df)
        
        # Normalize each split
        train_normalized = self._normalize(train_df, 'train')
        val_normalized = self._normalize(val_df, 'validation')
        test_normalized = self._normalize(test_df, 'test')
        
        # Generate sequences within each split
        X_train = self._generate_sequences(train_normalized, 'train')
        X_val = self._generate_sequences(val_normalized, 'validation')
        X_test = self._generate_sequences(test_normalized, 'test')
        
        print(f"\nSequence generation complete:")
        print(f"  Train sequences: {len(X_train)}")
        print(f"  Val sequences: {len(X_val)}")
        print(f"  Test sequences: {len(X_test)}")
        print(f"  Sequence shape: {X_train.shape[1:] if len(X_train) > 0 else 'N/A'}")
        
        return X_train, X_val, X_test
    
    def _handle_warmup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle Phase 2 warm-up NaNs.
        
        Strategy: Remove only the initial row with NaN (index 0).
        """
        print("\nHandling Phase 2 warm-up NaNs...")
        
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        
        if len(nan_cols) == 0:
            print("  ✓ No NaN values found")
            self.metadata['warmup_rows_removed'] = 0
            self.metadata['clean_rows'] = len(df)
            return df
        
        print(f"  NaN columns: {list(nan_cols.index)}")
        
        # Warm-up = the leading rows that contain any NaN (rolling-window /
        # difference start-up). Everything after must be NaN-free.
        valid = df.notna().all(axis=1).to_numpy()
        if not valid.any():
            raise ValueError("Every row contains NaN; cannot remove warm-up.")
        n_warmup = int(np.argmax(valid))
        df_clean = df.iloc[n_warmup:].reset_index(drop=True)
        
        remaining_nans = df_clean.isna().sum().sum()
        if remaining_nans > 0:
            raise ValueError(f"Unexpected NaNs remain after warm-up removal: {remaining_nans}")
        
        print(f"  ✓ Removed {n_warmup} warm-up row(s)")
        print(f"  ✓ Remaining data: {len(df_clean)} rows")
        
        self.metadata['warmup_rows_removed'] = n_warmup
        self.metadata['clean_rows'] = len(df_clean)
        
        return df_clean
    
    def _chronological_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically (NO shuffling).
        
        Returns:
            (train_df, val_df, test_df)
        """
        print(f"\nChronological split...")
        
        n = len(df)
        
        # Calculate split indices
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        # Split chronologically
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        # Extract timestamps
        train_time = (train_df['Time'].min(), train_df['Time'].max())
        val_time = (val_df['Time'].min(), val_df['Time'].max())
        test_time = (test_df['Time'].min(), test_df['Time'].max())
        
        print(f"  Train: {len(train_df)} rows | Time: [{train_time[0]:.2f}, {train_time[1]:.2f}]s")
        print(f"  Val:   {len(val_df)} rows | Time: [{val_time[0]:.2f}, {val_time[1]:.2f}]s")
        print(f"  Test:  {len(test_df)} rows | Time: [{test_time[0]:.2f}, {test_time[1]:.2f}]s")
        
        # Verify no overlap
        assert train_time[1] < val_time[0], "Train/Val overlap detected!"
        assert val_time[1] < test_time[0], "Val/Test overlap detected!"
        
        print(f"  ✓ No temporal overlap between splits")
        
        # Store metadata
        self.metadata['train_rows'] = len(train_df)
        self.metadata['val_rows'] = len(val_df)
        self.metadata['test_rows'] = len(test_df)
        self.metadata['train_time_range'] = train_time
        self.metadata['val_time_range'] = val_time
        self.metadata['test_time_range'] = test_time
        
        return train_df, val_df, test_df
    
    def _fit_scaler(self, train_df: pd.DataFrame):
        """
        Fit scaler on TRAINING data only.
        
        CRITICAL: Never fit on validation or test data!
        """
        print(f"\nFitting scaler on TRAINING data only...")
        
        # Extract features (exclude Time)
        train_features = train_df[self.feature_names].values
        
        # Fit StandardScaler
        self.scaler = StandardScaler()
        self.scaler.fit(train_features)
        
        print(f"  ✓ Scaler fitted on {len(train_features)} training samples")
        print(f"  ✓ Feature dim: {train_features.shape[1]}")
        
        self.metadata['scaler_type'] = 'StandardScaler'
        self.metadata['scaler_fitted_on'] = 'training_data_only'
    
    def _normalize(self, df: pd.DataFrame, split_name: str) -> np.ndarray:
        """
        Normalize features using the FITTED scaler.
        
        Args:
            df: DataFrame to normalize
            split_name: 'train', 'validation', or 'test'
        
        Returns:
            Normalized feature array
        """
        features = df[self.feature_names].values
        normalized = self.scaler.transform(features)
        
        print(f"  {split_name.capitalize()}: Normalized {len(features)} samples using fitted scaler")
        
        return normalized
    
    def _generate_sequences(self, data: np.ndarray, split_name: str) -> np.ndarray:
        """
        Generate sliding window sequences.
        
        IMPORTANT: Sequences do NOT cross split boundaries.
        
        Args:
            data: Normalized feature array (n_samples, n_features)
            split_name: 'train', 'validation', or 'test'
        
        Returns:
            Sequences array (n_sequences, sequence_length, n_features)
        """
        n_samples, n_features = data.shape
        
        # Calculate number of sequences
        if n_samples < self.sequence_length:
            print(f"  ⚠ {split_name}: Insufficient samples ({n_samples}) for sequence length ({self.sequence_length})")
            return np.array([])
        
        n_sequences = (n_samples - self.sequence_length) // self.stride + 1
        
        # Generate sequences
        sequences = []
        for i in range(n_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + self.sequence_length
            
            if end_idx > n_samples:
                break
            
            sequence = data[start_idx:end_idx]
            sequences.append(sequence)
        
        sequences = np.array(sequences)
        
        print(f"  {split_name.capitalize()}: Generated {len(sequences)} sequences")
        
        # Store metadata
        self.metadata[f'{split_name}_sequences'] = len(sequences)
        
        return sequences
    
    def save_outputs(self, output_dir: str = 'data/sequences'):
        """
        Save sequences, scaler, and metadata.
        
        Args:
            output_dir: Directory to save sequences
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving outputs to {output_path}...")
        
        # Save metadata
        metadata_path = output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            # Convert tuples to lists for JSON serialization
            meta_copy = self.metadata.copy()
            for key in ['train_time_range', 'val_time_range', 'test_time_range']:
                if key in meta_copy:
                    meta_copy[key] = list(meta_copy[key])
            
            json.dump(meta_copy, f, indent=2)
        print(f"  ✓ Metadata: {metadata_path}")
        
        # Save scaler
        scaler_path = REPO_ROOT / 'models' / 'feature_scaler.pkl'
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  ✓ Scaler: {scaler_path}")
        
        # Compute and save training envelope (Task 1a)
        print("\nComputing training envelope from TRAINING rows...")
        import training_envelope as te
        # Reconstruct cleaned_telemetry path from features path
        features_path = Path(self.metadata['input_file'])
        cleaned_path = features_path.parent.parent / 'processed' / 'cleaned_telemetry.csv'
        envelope = te.compute_training_envelope(str(cleaned_path), self.metadata, _cc.get_physical_channels())
        te.save_training_envelope(envelope, str(REPO_ROOT / 'models' / 'training_envelope.json'))
    
    def save_sequences(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, 
                      output_dir: str = 'data/sequences'):
        """
        Save sequence arrays to disk.
        
        Args:
            X_train, X_val, X_test: Sequence arrays
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy arrays
        if len(X_train) > 0:
            np.save(output_path / 'X_train.npy', X_train)
            print(f"  ✓ Train sequences: {output_path / 'X_train.npy'}")
        
        if len(X_val) > 0:
            np.save(output_path / 'X_val.npy', X_val)
            print(f"  ✓ Val sequences: {output_path / 'X_val.npy'}")
        
        if len(X_test) > 0:
            np.save(output_path / 'X_test.npy', X_test)
            print(f"  ✓ Test sequences: {output_path / 'X_test.npy'}")


def run_phase3(features_csv_path: str, output_dir: str = 'data/sequences') -> bool:
    """
    Run Phase 3 sequence preparation.
    
    Args:
        features_csv_path: Path to Phase 2 features
        output_dir: Output directory for sequences
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print("="*80)
        print(" "*22 + "PHASE 3 - SEQUENCE PREPARATION")
        print("="*80)
        
        # Initialize preparator
        preparator = SequencePreparator()  # all parameters from pipeline_config
        
        # Prepare sequences
        X_train, X_val, X_test = preparator.prepare_sequences(features_csv_path)
        
        # Save outputs
        preparator.save_sequences(X_train, X_val, X_test, output_dir)
        preparator.save_outputs(output_dir)
        
        print("\n" + "="*80)
        print(" "*28 + "PHASE 3 COMPLETE")
        print("="*80)
        print(f"\nSequences prepared:")
        print(f"  Train: {len(X_train)} sequences")
        print(f"  Val: {len(X_val)} sequences")
        print(f"  Test: {len(X_test)} sequences")
        print(f"\nOutputs saved to:")
        print(f"  Sequences: {output_dir}/")
        print(f"  Scaler: {REPO_ROOT / 'models' / 'feature_scaler.pkl'}")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(" "*28 + "PHASE 3 FAILED")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for Phase 3."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prepare_sequences.py <features_csv>")
        sys.exit(1)
    
    features_csv = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/sequences'
    
    success = run_phase3(features_csv, output_dir)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
