> **Note:** channel lists, feature counts and parameter values in this file describe the original 10-channel prototype.
> The authoritative values are `channel_config.py`, `pipeline_config.py` and the top-level `README.md`.

# Phase 1: DAQ Telemetry Preprocessing

## Overview

Phase 1 validates and preprocesses raw DAQ telemetry data from the MALE UAV engine system. For this prototype, preprocessing focuses on correcting intentionally injected noise in **FuelFlow** and **TPS** channels while preserving all other channels unchanged.

## What Phase 1 Does

### 1. Data Validation
- Validates schema (all required columns present)
- Validates Time column (monotonicity, sampling rate calculation)
- Detects missing (NaN) values
- Detects infinite values
- Reports data quality metrics

### 2. FuelFlow Correction
**Why**: Fuel flow cannot be physically negative (impossible to have negative flow rate)

**What we do**:
- Detect all FuelFlow values < 0
- Flag them in quality_flags.csv
- Clip negative values to 0 in cleaned output
- Preserve original raw values unchanged

### 3. TPS Correction
**Why**: TPS (Throttle Position Sensor) is a percentage measurement with valid range [0, 100]

**What we do**:
- Detect TPS values < 0 or > 100
- Flag them in quality_flags.csv  
- Clip out-of-range values to [0, 100] in cleaned output
- Preserve original raw values unchanged

### 4. Other Channels (Preserved Unchanged)
The following channels are **NOT modified** in Phase 1:
- EngTrq (Engine Torque)
- RPM (Engine Speed)
- MAT (Manifold Air Temperature)
- MAP (Manifold Absolute Pressure)
- EGT (Exhaust Gas Temperature)
- CH (Cylinder Head Temperature)
- BaroPress (Barometric Pressure)
- TurboSpd (Turbocharger Speed)

**Why preserved**: These channels do not contain intentional noise injection in this prototype dataset. Future phases may apply validated filtering if needed.

## What Quality Flags Mean

The `quality_flags.csv` file contains quality codes for each sample and channel:

- **0 = Valid**: No issues detected, value used as-is
- **1 = Invalid/Corrected**: Value was flagged as invalid and corrected in cleaned output

Quality flags enable:
- Traceability: Know exactly which samples were modified
- Auditability: Can reconstruct which raw values were problematic
- Downstream awareness: Later analysis stages can weight/filter based on quality

## Why Raw Data is Preserved

The original raw CSV file is **never modified**. This ensures:
1. **Reproducibility**: Can re-run preprocessing with different parameters
2. **Auditability**: Can always inspect original measurements
3. **Safety**: No risk of data loss from preprocessing errors
4. **Comparison**: Can compare raw vs. cleaned to verify corrections

## Files Produced by Phase 1

### Outputs (in `data/processed/`):

1. **cleaned_telemetry.csv**
   - Preprocessed telemetry ready for feature extraction
   - FuelFlow and TPS corrected to valid ranges
   - All other channels preserved unchanged
   
2. **quality_flags.csv**
   - Quality codes for each sample (0=valid, 1=corrected)
   - Timestamp column for alignment
   - One quality column per telemetry channel

3. **preprocessing_report.json**
   - Complete statistics and metadata
   - Sampling rate, duration, row counts
   - Raw vs. cleaned statistics (min/max/mean/std)
   - Number and percentage of corrections per channel
   - Validation confirmation flags

## How to Run Phase 1

### Prerequisites
```bash
# Ensure you have Python 3.7+ with pandas and numpy
pip install pandas numpy
```

### Run Preprocessing
```bash
cd integrated_pipeline
python phase1_preprocessing/preprocess.py "data/raw/Teammate_Telemetry_Data (1).csv"
```

### Run Validation
```bash
cd integrated_pipeline
python phase1_preprocessing/validate.py "data/raw/Teammate_Telemetry_Data (1).csv"
```

The validation script checks:
- ✓ Raw file not modified
- ✓ Row counts match (no silent deletions)
- ✓ FuelFlow all >= 0 in cleaned data
- ✓ TPS in [0, 100] in cleaned data
- ✓ All other channels unchanged
- ✓ Quality flags match actual corrections
- ✓ No unexpected columns added
- ✓ Statistics report complete

## Expected Results

For the prototype dataset (~50 Hz telemetry, ~36,000 samples):

**FuelFlow**:
- Raw range includes negative values (injected noise)
- Invalid samples: ~50% (intentional)
- Cleaned: All values >= 0

**TPS**:
- Raw range includes values outside [0, 100]
- Invalid samples: ~28-30% (intentional)
- Cleaned: All values in [0, 100]

**Other channels**:
- 0 samples modified (preserved unchanged)

## Important Notes

### This is a Prototype
- Preprocessing rules are specific to this prototype dataset
- FuelFlow and TPS corrections are based on known noise injection
- **NOT validated for real aircraft operation**
- Real deployment would require additional validation with domain experts

### No Over-Processing
Phase 1 intentionally does **NOT**:
- Apply Hampel/MAD filtering to all channels
- Apply EMA smoothing
- Apply Savitzky-Golay filtering
- Apply Butterworth filtering (that's for vibration data, different freq regime)
- Clip MAP or other channels to arbitrary limits
- Remove genuine engine transients

**Why**: Avoid removing real signal features. Only correct known invalid ranges.

### Next Phases (Not Yet Implemented)
- **Phase 2**: Feature extraction (time/frequency domain)
- **Phase 3**: Sequence preparation + chronological splitting + train-only normalization
- **Phase 4**: LSTM autoencoder training
- **Phase 5**: Anomaly detection via reconstruction error

## Questions or Issues?

If preprocessing produces unexpected results:
1. Check `preprocessing_report.json` for statistics
2. Run `validate.py` to verify correctness
3. Inspect quality_flags.csv to see which samples were flagged
4. Compare raw vs. cleaned CSVs manually for specific channels

**DO NOT proceed to Phase 2** until Phase 1 outputs are reviewed and approved.
