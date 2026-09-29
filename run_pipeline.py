"""
Integrated UAV Engine Telemetry Processing Pipeline

Main entry point for the integrated pipeline.
Supports Phases 1-5.

Usage:
    python run_pipeline.py --mode <mode> --input <csv_path> [--output <dir>]

Modes:
    preprocess - Phase 1: Data validation and preprocessing
    features   - Phases 1+2: Preprocessing + Feature extraction
    sequences  - Phases 1+2+3: Full pipeline through sequence preparation
    train      - Phases 1+2+3+4: Full pipeline + LSTM Autoencoder training
    detect     - Phases 1+2+3+4+5: Full pipeline + Anomaly detection
    inference  - Inference using existing trained artifacts (NO retraining)
"""

import argparse
import sys
from pathlib import Path

# Add phase modules to path
sys.path.insert(0, str(Path(__file__).parent / 'phase1_preprocessing'))
sys.path.insert(0, str(Path(__file__).parent / 'phase2_feature_extraction'))
sys.path.insert(0, str(Path(__file__).parent / 'phase3_sequence_preparation'))
sys.path.insert(0, str(Path(__file__).parent / 'phase4_autoencoder'))
sys.path.insert(0, str(Path(__file__).parent / 'phase5_anomaly_detection'))

from preprocess import TelemetryPreprocessor
from extract_features import run_phase2
from prepare_sequences import run_phase3
from train_autoencoder import train_autoencoder
from detect_anomalies import run_anomaly_detection
sys.path.insert(0, str(Path(__file__).parent))
import pipeline_config as pc


def run_phase1_preprocess(input_csv: str, output_dir: str = None) -> tuple:
    """Run Phase 1 preprocessing on any input CSV."""
    if output_dir is None:
        output_dir = 'data/processed'
    
    input_path = Path(input_csv)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return False, None
    
    print(f"Pipeline Mode: PREPROCESS (Phase 1)")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Training-side call: FIT the filter cutoffs on this data (saved to models/).
    preprocessor = TelemetryPreprocessor(str(input_path), fit_filters=True)
    success = preprocessor.run(str(output_path))
    cleaned_csv = output_path / 'cleaned_telemetry.csv'
    
    if success:
        print(f"\n✓ Phase 1 preprocessing complete")
        print(f"  Outputs saved to: {output_path}")
        print(f"  Cleaned CSV: {cleaned_csv}")
        return True, str(cleaned_csv)
    else:
        return False, None


def run_phase1_and_phase2(input_csv: str, output_dir: str = None) -> tuple:
    """Run Phase 1 preprocessing followed by Phase 2 feature extraction."""
    print("="*80)
    print(" "*20 + "INTEGRATED PIPELINE: PHASES 1 + 2")
    print("="*80)
    print()
    
    if output_dir is None:
        preprocess_dir = 'data/processed'
        features_dir = 'data/features'
    else:
        base_path = Path(output_dir)
        preprocess_dir = str(base_path / 'processed')
        features_dir = str(base_path / 'features')
    
    # Run Phase 1
    print("[PHASE 1] Running preprocessing...")
    print("-"*80)
    phase1_success, cleaned_csv = run_phase1_preprocess(input_csv, preprocess_dir)
    
    if not phase1_success:
        print("\n✗ Phase 1 failed - cannot proceed to Phase 2")
        return False, None
    
    print()
    print("="*80)
    
    # Run Phase 2
    print("[PHASE 2] Running feature extraction...")
    print("-"*80)
    phase2_success = run_phase2(cleaned_csv, features_dir)
    
    if not phase2_success:
        print("\n✗ Phase 2 failed")
        return False, None
    
    features_csv = Path(features_dir) / 'extracted_features.csv'
    
    print("\n" + "="*80)
    print(" "*25 + "PIPELINE COMPLETE")
    print("="*80)
    print(f"\nSuccessfully completed:")
    print(f"  ✓ Phase 1: Preprocessing")
    print(f"  ✓ Phase 2: Feature Extraction")
    print(f"\nOutputs:")
    print(f"  Preprocessing: {preprocess_dir}/")
    print(f"  Features: {features_dir}/")
    
    return True, str(features_csv)


def run_full_pipeline(input_csv: str, output_dir: str = None) -> bool:
    """Run Phases 1+2+3 (full pipeline through sequence preparation)."""
    print("="*80)
    print(" "*18 + "INTEGRATED PIPELINE: PHASES 1 + 2 + 3")
    print("="*80)
    print()
    
    if output_dir is None:
        preprocess_dir = 'data/processed'
        features_dir = 'data/features'
        sequences_dir = 'data/sequences'
    else:
        base_path = Path(output_dir)
        preprocess_dir = str(base_path / 'processed')
        features_dir = str(base_path / 'features')
        sequences_dir = str(base_path / 'sequences')
    
    # Run Phases 1+2
    phase12_success, features_csv = run_phase1_and_phase2(input_csv, output_dir)
    
    if not phase12_success:
        return False
    
    print()
    print("="*80)
    
    # Run Phase 3
    print("[PHASE 3] Running sequence preparation...")
    print("-"*80)
    phase3_success = run_phase3(features_csv, sequences_dir)
    
    if not phase3_success:
        print("\n✗ Phase 3 failed")
        return False
    
    print("\n" + "="*80)
    print(" "*25 + "PIPELINE COMPLETE")
    print("="*80)
    print(f"\nSuccessfully completed:")
    print(f"  ✓ Phase 1: Preprocessing")
    print(f"  ✓ Phase 2: Feature Extraction")
    print(f"  ✓ Phase 3: Sequence Preparation")
    print(f"\nOutputs:")
    print(f"  Preprocessing: {preprocess_dir}/")
    print(f"  Features: {features_dir}/")
    print(f"  Sequences: {sequences_dir}/")
    print(f"  Scaler: models/feature_scaler.pkl")
    
    return True


def run_pipeline_with_training(input_csv: str, 
                               output_dir: str = None,
                               epochs: int = None,
                               batch_size: int = None) -> bool:
    """Run Phases 1+2+3+4 (full pipeline + autoencoder training)."""
    print("="*80)
    print(" "*15 + "INTEGRATED PIPELINE: PHASES 1 + 2 + 3 + 4")
    print("="*80)
    print()
    
    if output_dir is None:
        base_dir = '.'
        sequences_dir = 'data/sequences'
    else:
        base_dir = output_dir
        sequences_dir = str(Path(output_dir) / 'data' / 'sequences')
    
    # Run Phases 1+2+3
    phase123_success = run_full_pipeline(input_csv, output_dir)
    
    if not phase123_success:
        return False
    
    print()
    print("="*80)
    
    # Run Phase 4
    print("[PHASE 4] Training LSTM autoencoder...")
    print("-"*80)
    
    try:
        results = train_autoencoder(
            sequences_dir=sequences_dir,
            output_dir=base_dir,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print("\n" + "="*80)
        print(" "*25 + "PIPELINE COMPLETE")
        print("="*80)
        print(f"\nSuccessfully completed:")
        print(f"  ✓ Phase 1: Preprocessing")
        print(f"  ✓ Phase 2: Feature Extraction")
        print(f"  ✓ Phase 3: Sequence Preparation")
        print(f"  ✓ Phase 4: LSTM Autoencoder Training")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Phase 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pipeline_with_detection(input_csv: str,
                                output_dir: str = None,
                                epochs: int = None,
                                batch_size: int = None,
                                percentile: float = None) -> bool:
    """Run Phases 1+2+3+4+5 (full pipeline + training + detection)."""
    print("="*80)
    print(" "*12 + "INTEGRATED PIPELINE: PHASES 1 + 2 + 3 + 4 + 5")
    print("="*80)
    print()
    
    if output_dir is None:
        base_dir = '.'
    else:
        base_dir = output_dir
    
    # Run Phases 1+2+3+4
    phase1234_success = run_pipeline_with_training(input_csv, output_dir, epochs, batch_size)
    
    if not phase1234_success:
        return False
    
    print()
    print("="*80)
    
    # Run Phase 5
    print("[PHASE 5] Running anomaly detection...")
    print("-"*80)
    
    try:
        results = run_anomaly_detection(base_dir, percentile)
        
        print("\n" + "="*80)
        print(" "*25 + "PIPELINE COMPLETE")
        print("="*80)
        print(f"\nSuccessfully completed:")
        print(f"  ✓ Phase 1: Preprocessing")
        print(f"  ✓ Phase 2: Feature Extraction")
        print(f"  ✓ Phase 3: Sequence Preparation")
        print(f"  ✓ Phase 4: LSTM Autoencoder Training")
        print(f"  ✓ Phase 5: Anomaly Detection")
        print(f"\nDetection Results:")
        print(f"  Threshold: {results['threshold_calculation']['threshold_value']:.6f}")
        print(f"  Normal: {results['classification_results']['normal_count']}")
        print(f"  Anomaly: {results['classification_results']['anomaly_count']}")
        print(f"  Anomaly %: {results['classification_results']['anomaly_percentage']:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Phase 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return False



def run_inference_mode(input_csv: str, output_dir: str = None) -> bool:
    """Run inference using existing trained artifacts (NO retraining/refit/recalc).
    The implementation lives in inference_core.py (shared with phase 5)."""
    import inference_core
    return inference_core.run_inference(input_csv, base_dir=Path(__file__).resolve().parent) is not None


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='UAV Engine Telemetry Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        required=True,
        choices=['preprocess', 'features', 'sequences', 'train', 'detect', 'inference'],
        help='Pipeline mode to execute'
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--output',
        default=None,
        help='Output directory (default: data/)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Maximum training epochs (default: EPOCHS in pipeline_config; train/detect modes only)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Training batch size (default: BATCH_SIZE in pipeline_config; train/detect modes only)'
    )
    
    parser.add_argument(
        '--percentile',
        type=float,
        default=None,
        help='Anomaly threshold percentile (default: THRESHOLD_PERCENTILE in pipeline_config; detect mode only)'
    )
    
    args = parser.parse_args()
    
    # Route to appropriate phase
    if args.mode == 'preprocess':
        success, _ = run_phase1_preprocess(args.input, args.output)
        sys.exit(0 if success else 1)
    
    elif args.mode == 'features':
        success, _ = run_phase1_and_phase2(args.input, args.output)
        sys.exit(0 if success else 1)
    
    elif args.mode == 'sequences':
        success = run_full_pipeline(args.input, args.output)
        sys.exit(0 if success else 1)
    
    elif args.mode == 'train':
        success = run_pipeline_with_training(
            args.input,
            args.output,
            args.epochs,
            args.batch_size
        )
        sys.exit(0 if success else 1)
    
    elif args.mode == 'detect':
        success = run_pipeline_with_detection(
            args.input,
            args.output,
            args.epochs,
            args.batch_size,
            args.percentile
        )
        sys.exit(0 if success else 1)
    
    elif args.mode == 'inference':
        success = run_inference_mode(args.input, args.output)
        sys.exit(0 if success else 1)
    
    else:
        print(f"ERROR: Mode '{args.mode}' not yet implemented")
        sys.exit(1)


if __name__ == '__main__':
    main()
