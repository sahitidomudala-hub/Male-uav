> **Note:** channel lists, feature counts and parameter values in this file describe the original 10-channel prototype.
> The authoritative values are `channel_config.py`, `pipeline_config.py` and the top-level `README.md`.

# Phase 4: LSTM Autoencoder

## Status

✓ Model trained and saved
✓ Reconstruction errors generated and verified
⚠ Training history unavailable (interrupted run)

## Model

File: `models/lstm_autoencoder.keras`
- Input: (None, 50, 24)
- Output: (None, 50, 24)
- Parameters: 57,368
- Backend: Keras 3 + JAX

Architecture:
```
Input (50, 24)
  ↓
LSTM Encoder (64 units)
  ↓
Latent (64)
  ↓
RepeatVector (50)
  ↓
LSTM Decoder (64 units)
  ↓
TimeDistributed Dense (24)
  ↓
Output (50, 24)
```

## Reconstruction Errors

**Validation set (5,377 sequences):**
- File: `outputs/reconstruction_errors_val.csv`
- Verified to match current model predictions

**Test set (5,379 sequences):**
- File: `outputs/reconstruction_errors_test.csv`
- Verified to match current model predictions

Errors calculated as MSE per sequence: mean((input - reconstruction)², axis=(timesteps, features))

## Training History

Complete epoch-by-epoch training history is **not available**.

The original training run was interrupted before the Keras History object could be saved. The model checkpoint was preserved via the ModelCheckpoint callback during training.

**What we have:**
- ✓ Trained model (checkpoint saved during training)
- ✓ Model architecture verified
- ✓ Reconstruction errors verified against current model

**What we don't have:**
- ✗ Epoch-by-epoch loss progression
- ✗ Final training/validation loss values
- ✗ Number of epochs completed

The model is functional and produces valid reconstruction errors.

## Usage

### Load Model
```python
import os
os.environ['KERAS_BACKEND'] = 'jax'
import keras

model = keras.models.load_model('models/lstm_autoencoder.keras')
```

### Generate Reconstructions
```python
import numpy as np
import pickle

# Load scaler
with open('models/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Process new data through Phases 1-3
# Then predict
reconstructed = model.predict(sequences, batch_size=64)
errors = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))
```

## Validation

Run: `python phase4_autoencoder/validate.py`

All checks except training history verification pass.

## Scientific Honesty

✓ Test data NOT used for training
✓ NO synthetic data generated
✓ Reconstruction errors are real model outputs
✓ Model architecture verified
✓ No fabricated training history

## Limitations

- Training history unavailable (interruption during original run)
- Cannot report final training/validation losses
- Cannot report total epochs completed
- Model performance must be evaluated through reconstruction errors only

## Next: Phase 5

Use reconstruction errors for anomaly detection via threshold-based classification.
