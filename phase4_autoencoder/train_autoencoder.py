"""
Phase 4: LSTM Autoencoder Training
===================================

Trains LSTM autoencoder to reconstruct normal engine telemetry sequences.
Uses Phase 3 sequences: (n_sequences, sequence_length, n_features)

Architecture:
    Input → LSTM Encoder → Latent → RepeatVector → LSTM Decoder → Output

Training:
    - Train on X_train (reconstruction task: X_train → X_train)
    - Validate on X_val
    - DO NOT use X_test for training/tuning
    
Output:
    - Trained model (.keras format)
    - Training history
    - Model metadata
    - Reconstruction errors (val and test)
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'  # Use TensorFlow backend (more stable in this environment)

import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import pipeline_config as _pc
import schema_guard as _sg

try:
    import keras
    from keras import layers, models, callbacks
    KERAS_AVAILABLE = True
except ImportError as _e:
    KERAS_AVAILABLE = False
    warnings.warn(f"Keras not available ({_e}). Install TensorFlow (pip install tensorflow) in a Python version it supports.")


class LSTMAutoencoder:
    """LSTM Autoencoder for temporal sequence reconstruction."""
    
    def __init__(self,
                 sequence_length: int,
                 n_features: int,
                 encoder_units: int = None,
                 decoder_units: int = None,
                 learning_rate: float = None):
        """
        Initialize LSTM Autoencoder.
        
        Args:
            sequence_length: Number of timesteps per sequence (50 @ 50Hz = 1 second)
            n_features: Number of features per timestep (24 from Phase 2)
            encoder_units: LSTM units in encoder layer
            decoder_units: LSTM units in decoder layer
            learning_rate: Learning rate for Adam optimizer
        """
        if not KERAS_AVAILABLE:
            raise ImportError("Keras required. Install TensorFlow: pip install tensorflow")
        encoder_units = _pc.LSTM_UNITS if encoder_units is None else encoder_units
        decoder_units = _pc.LSTM_UNITS if decoder_units is None else decoder_units
        learning_rate = _pc.LEARNING_RATE if learning_rate is None else learning_rate
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.learning_rate = learning_rate
        
        self.model = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """
        Build LSTM Autoencoder architecture.
        
        Returns:
            Compiled Keras model
        """
        print("="*70)
        print("BUILDING LSTM AUTOENCODER")
        print("="*70)
        print(f"Input shape: (batch, {self.sequence_length}, {self.n_features})")
        print(f"Encoder LSTM units: {self.encoder_units}")
        print(f"Decoder LSTM units: {self.decoder_units}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Keras backend: {keras.backend.backend()}")
        
        # INPUT
        inputs = layers.Input(
            shape=(self.sequence_length, self.n_features),
            name='input_sequence'
        )
        
        # ENCODER
        # Compress temporal sequence into fixed-size latent representation
        encoded = layers.LSTM(
            self.encoder_units,
            activation='tanh',
            return_sequences=False,  # Return only final hidden state
            name='encoder_lstm'
        )(inputs)
        # Shape: (batch, encoder_units)
        
        # REPEAT LATENT REPRESENTATION
        # Replicate latent vector for each timestep
        repeated = layers.RepeatVector(
            self.sequence_length,
            name='repeat_latent'
        )(encoded)
        # Shape: (batch, sequence_length, encoder_units)
        
        # DECODER
        # Reconstruct sequence from latent representation
        decoded = layers.LSTM(
            self.decoder_units,
            activation='tanh',
            return_sequences=True,  # Return full sequence
            name='decoder_lstm'
        )(repeated)
        # Shape: (batch, sequence_length, decoder_units)
        
        # OUTPUT RECONSTRUCTION
        # TimeDistributed applies Dense to each timestep independently
        outputs = layers.TimeDistributed(
            layers.Dense(self.n_features, activation='linear'),
            name='reconstructed_sequence'
        )(decoded)
        # Shape: (batch, sequence_length, n_features) - same as input
        
        # BUILD MODEL
        model = models.Model(inputs=inputs, outputs=outputs, name='lstm_autoencoder')
        
        # COMPILE
        optimizer = keras.optimizers.get({'class_name': _pc.OPTIMIZER, 'config': {'learning_rate': self.learning_rate}})
        model.compile(
            optimizer=optimizer,
            loss=_pc.LOSS,           # reconstruction loss (pipeline_config)
            metrics=list(_pc.METRICS)
        )
        
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE")
        print("="*70)
        model.summary()
        print("="*70)
        
        self.model = model
        return model
    
    def train(self,
              X_train: np.ndarray,
              X_val: np.ndarray,
              epochs: int = None,
              batch_size: int = None,
              model_save_path: Optional[str] = None) -> Dict:
        """
        Train autoencoder on training sequences.
        
        Args:
            X_train: Training sequences (n_train, sequence_length, n_features)
            X_val: Validation sequences (n_val, sequence_length, n_features)
            epochs: Maximum training epochs
            batch_size: Training batch size
            model_save_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
        epochs = _pc.EPOCHS if epochs is None else epochs
        batch_size = _pc.BATCH_SIZE if batch_size is None else batch_size
        if self.model is None:
            self.build_model()
        
        print("\n" + "="*70)
        print("TRAINING LSTM AUTOENCODER")
        print("="*70)
        print(f"Training sequences: {X_train.shape[0]}")
        print(f"Validation sequences: {X_val.shape[0]}")
        print(f"Sequence shape: ({X_train.shape[1]}, {X_train.shape[2]})")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {epochs}")
        
        # Prepare callbacks
        callback_list = []
        
        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=_pc.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)
        print(f"  ✓ Early stopping: patience={_pc.EARLY_STOPPING_PATIENCE}")
        
        # Model checkpointing
        if model_save_path is not None:
            checkpoint = callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callback_list.append(checkpoint)
            print(f"  ✓ Checkpointing: {model_save_path}")
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=_pc.LR_REDUCTION_FACTOR,
            patience=_pc.LR_REDUCTION_PATIENCE,
            min_lr=_pc.LR_MIN,
            verbose=1
        )
        callback_list.append(reduce_lr)
        print(f"  ✓ Learning rate reduction: factor={_pc.LR_REDUCTION_FACTOR}, patience={_pc.LR_REDUCTION_PATIENCE}")
        
        print("\n" + "="*70)
        print("TRAINING IN PROGRESS...")
        print("="*70)
        
        # TRAIN (autoencoder: input = output = same sequences)
        history = self.model.fit(
            X_train, X_train,  # Reconstruction task
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.history = history.history
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Epochs completed: {len(history.history['loss'])}")
        print(f"Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
        print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
        print("="*70)
        
        return self.history
    
    def predict(self, sequences: np.ndarray, batch_size: int = None) -> np.ndarray:
        """
        Reconstruct sequences using trained model.
        
        Args:
            sequences: Input sequences (n_sequences, sequence_length, n_features)
            batch_size: Prediction batch size
            
        Returns:
            Reconstructed sequences (same shape as input)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        batch_size = _pc.INFERENCE_BATCH_SIZE if batch_size is None else batch_size
        reconstructed = self.model.predict(sequences, batch_size=batch_size, verbose=0)
        return reconstructed
    
    def calculate_reconstruction_errors(self,
                                        sequences: np.ndarray,
                                        batch_size: int = None) -> np.ndarray:
        """
        Calculate per-sequence reconstruction error (MSE).
        
        Args:
            sequences: Input sequences (n_sequences, sequence_length, n_features)
            batch_size: Prediction batch size
            
        Returns:
            Reconstruction errors (n_sequences,) - MSE per sequence
        """
        reconstructed = self.predict(sequences, batch_size=batch_size)  # None -> INFERENCE_BATCH_SIZE
        
        # Calculate MSE per sequence: average over (timesteps, features)
        errors = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))
        
        return errors
    
    def save_model(self, save_path: str):
        """
        Save trained model.
        
        Args:
            save_path: Path to save model (.keras format)
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(save_path)
        print(f"Model saved: {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load trained model.
        
        Args:
            load_path: Path to saved model (.keras format)
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        self.model = keras.models.load_model(load_path)
        print(f"Model loaded: {load_path}")
        print(f"  Input shape: {self.model.input_shape}")
        print(f"  Output shape: {self.model.output_shape}")
    
    def get_config(self) -> Dict:
        """
        Get model configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'encoder_units': self.encoder_units,
            'decoder_units': self.decoder_units,
            'learning_rate': self.learning_rate
        }


def train_autoencoder(sequences_dir: str,
                     output_dir: str,
                     epochs: int = None,
                     batch_size: int = None) -> Dict:
    """
    Main training function for Phase 4.
    
    Args:
        sequences_dir: Directory containing Phase 3 sequences
        output_dir: Directory to save model and outputs
        epochs: Maximum training epochs
        batch_size: Training batch size
        
    Returns:
        Training results dictionary
    """
    print("="*70)
    print("PHASE 4: LSTM AUTOENCODER TRAINING")
    print("="*70)
    
    epochs = _pc.EPOCHS if epochs is None else epochs
    batch_size = _pc.BATCH_SIZE if batch_size is None else batch_size
    sequences_dir = Path(sequences_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("\n[1/7] Loading Phase 3 metadata...")
    metadata_path = sequences_dir / 'metadata.json'
    with open(metadata_path, 'r', encoding='utf-8-sig') as f:
        metadata = json.load(f)

    # The sequences on disk must have been produced with the CURRENT config.
    _sg.verify_schema(str(output_dir))

    sequence_length = metadata['sequence_length']  # from Phase 3, not a literal
    n_features = metadata['num_features']
    n_train = metadata['train_sequences']
    n_val = metadata['validation_sequences']
    n_test = metadata['test_sequences']
    
    print(f"  Sequence shape: ({sequence_length}, {n_features})")
    print(f"  Train sequences: {n_train}")
    print(f"  Validation sequences: {n_val}")
    print(f"  Test sequences: {n_test}")
    
    # Load sequences
    print("\n[2/7] Loading sequences...")
    X_train = np.load(sequences_dir / 'X_train.npy')
    X_val = np.load(sequences_dir / 'X_val.npy')
    X_test = np.load(sequences_dir / 'X_test.npy')
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Verify no NaN/Inf
    assert not np.any(np.isnan(X_train)), "X_train contains NaN"
    assert not np.any(np.isinf(X_train)), "X_train contains Inf"
    assert not np.any(np.isnan(X_val)), "X_val contains NaN"
    assert not np.any(np.isinf(X_val)), "X_val contains Inf"
    print("  ✓ No NaN/Inf in sequences")
    
    # Build model
    print("\n[3/7] Building model...")
    autoencoder = LSTMAutoencoder(
        sequence_length=sequence_length,
        n_features=n_features,
        encoder_units=_pc.LSTM_UNITS,
        decoder_units=_pc.LSTM_UNITS,
        learning_rate=_pc.LEARNING_RATE
    )
    autoencoder.build_model()
    
    # Count trainable parameters
    total_params = sum([np.prod(w.shape) for w in autoencoder.model.trainable_weights])
    print(f"\n  Total trainable parameters: {total_params:,}")
    
    # Train
    print("\n[4/7] Training model...")
    # Canonical filename: this is what inference and Phase 5 load.
    model_filename = 'lstm_autoencoder.keras'
    model_path = output_dir / 'models' / model_filename
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Model will be saved as: {model_filename}")
    
    history = autoencoder.train(
        X_train=X_train,
        X_val=X_val,
        epochs=epochs,
        batch_size=batch_size,
        model_save_path=str(model_path)
    )
    
    # Save final model (if checkpoint didn't save it)
    print("\n[5/7] Saving model...")
    autoencoder.save_model(str(model_path))
    
    # Save training history using csv module
    print("\n[6/7] Saving training history...")
    history_path = output_dir / 'outputs' / 'training_history.csv'
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all keys from history dict
    keys = list(history.keys())
    num_epochs = len(history[keys[0]])
    
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(keys)
        # Write data rows (each epoch)
        for epoch_idx in range(num_epochs):
            row = [history[key][epoch_idx] for key in keys]
            writer.writerow(row)
    
    print(f"  Training history saved: {history_path}")
    
    # Calculate reconstruction errors
    print("\n[7/7] Calculating reconstruction errors...")
    
    # Validation errors
    print("  Computing validation reconstruction errors...")
    val_errors = autoencoder.calculate_reconstruction_errors(X_val)
    val_errors_path = output_dir / 'outputs' / 'reconstruction_errors_val.csv'
    
    with open(val_errors_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence_index', 'reconstruction_error_mse'])
        for idx, error in enumerate(val_errors):
            writer.writerow([idx, error])
    
    print(f"    Saved: {val_errors_path}")
    print(f"    Min error: {val_errors.min():.6f}")
    print(f"    Max error: {val_errors.max():.6f}")
    print(f"    Mean error: {val_errors.mean():.6f}")
    print(f"    Median error: {np.median(val_errors):.6f}")
    
    # Test errors (for later Phase 5 use - NOT for training/tuning)
    print("\n  Computing test reconstruction errors...")
    test_errors = autoencoder.calculate_reconstruction_errors(X_test)
    test_errors_path = output_dir / 'outputs' / 'reconstruction_errors_test.csv'
    
    with open(test_errors_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence_index', 'reconstruction_error_mse'])
        for idx, error in enumerate(test_errors):
            writer.writerow([idx, error])
    
    print(f"    Saved: {test_errors_path}")
    print(f"    Min error: {test_errors.min():.6f}")
    print(f"    Max error: {test_errors.max():.6f}")
    print(f"    Mean error: {test_errors.mean():.6f}")
    print(f"    Median error: {np.median(test_errors):.6f}")
    
    # Verify no NaN/Inf in errors
    assert not np.any(np.isnan(val_errors)), "Validation errors contain NaN"
    assert not np.any(np.isinf(val_errors)), "Validation errors contain Inf"
    assert not np.any(np.isnan(test_errors)), "Test errors contain NaN"
    assert not np.any(np.isinf(test_errors)), "Test errors contain Inf"
    print("\n  ✓ No NaN/Inf in reconstruction errors")
    
    # Save model metadata
    print("\n[8/8] Saving model metadata...")
    model_metadata = {
        'model_architecture': 'LSTM Autoencoder',
        'sequence_length': int(sequence_length),
        'n_features': int(n_features),
        'encoder_units': _pc.LSTM_UNITS,
        'decoder_units': _pc.LSTM_UNITS,
        'learning_rate': _pc.LEARNING_RATE,
        'optimizer': _pc.OPTIMIZER,
        'loss_function': _pc.LOSS,
        'batch_size': int(batch_size),
        'max_epochs': int(epochs),
        'actual_epochs': len(history['loss']),
        'early_stopping_patience': _pc.EARLY_STOPPING_PATIENCE,
        'lr_reduction_patience': _pc.LR_REDUCTION_PATIENCE,
        'config_fingerprint': metadata.get('config_fingerprint'),
        'total_trainable_parameters': int(total_params),
        'training_sequences': int(n_train),
        'validation_sequences': int(n_val),
        'test_sequences': int(n_test),
        'final_train_loss': float(history['loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'best_val_loss': float(min(history['val_loss'])),
        'test_data_used_for_training': False,
        'synthetic_data_generated': False,
        'keras_backend': keras.backend.backend(),
        'model_file': Path(model_path).as_posix(),
        'training_history_file': Path(history_path).as_posix(),
        'val_errors_file': Path(val_errors_path).as_posix(),
        'test_errors_file': Path(test_errors_path).as_posix(),
        'phase3_metadata_file': Path(metadata_path).as_posix()
    }
    
    # Save metadata with feature-specific filename
    metadata_filename = f'model_metadata_{n_features}f.json'
    metadata_path = output_dir / 'phase4_autoencoder' / metadata_filename
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Metadata will be saved as: {metadata_filename}")
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    print(f"  Model metadata saved: {metadata_path}")
    
    print("\n" + "="*70)
    print("PHASE 4 TRAINING COMPLETE")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Training history: {history_path}")
    print(f"Validation errors: {val_errors_path}")
    print(f"Test errors: {test_errors_path}")
    print("="*70)
    
    return model_metadata


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM Autoencoder (Phase 4)')
    parser.add_argument('--sequences-dir', type=str, required=True,
                        help='Directory containing Phase 3 sequences')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for model and results')
    parser.add_argument('--epochs', type=int, default=_pc.EPOCHS,
                        help=f'Maximum training epochs (default from pipeline_config: {_pc.EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=_pc.BATCH_SIZE,
                        help=f'Training batch size (default from pipeline_config: {_pc.BATCH_SIZE})')
    
    args = parser.parse_args()
    
    results = train_autoencoder(
        sequences_dir=args.sequences_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
