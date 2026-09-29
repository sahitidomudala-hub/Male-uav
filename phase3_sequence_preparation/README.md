> **Note:** channel lists, feature counts and parameter values in this file describe the original 10-channel prototype.
> The authoritative values are `channel_config.py`, `pipeline_config.py` and the top-level `README.md`.

# Phase 3: Sequence Preparation

## Purpose

Phase 3 prepares time-series sequences for LSTM autoencoder training with strict safeguards against data and temporal leakage.

**CRITICAL**: This phase implements the train/validation/test split and normalization with extreme care to prevent any form of data leakage.

## Input

**Source**: Phase 2 features (`data/features/extracted_features.csv`)

**Data**: 36,178 rows × 25 features @ 50 Hz

**Warm-up handling**: Removes first row (index 0) with NaN from Phase 2 moving statistics

## Core Design Principles

### 1. Chronological Split (NO Shuffling)
- Timeline split: 70% train / 15% val / 15% test
- Earliest data → training
- Middle data → validation
- Latest data → test
- **NO random shuffling** (would break temporal order)

### 2. Prevent Data Leakage
**CRITICAL RULE**: Scaler fitted ONLY on training data

**Correct procedure**:
1. Split timeline chronologically FIRST
2. Fit StandardScaler on training features ONLY
3. Transform training using fitted scaler
4. Transform validation using SAME fitted scaler
5. Transform test using SAME fitted scaler

**NEVER**:
- ❌ Fit scaler on entire dataset before splitting
- ❌ Fit separate scalers for val/test
- ❌ Use future data for training normalization

### 3. Prevent Temporal Leakage
**CRITICAL RULE**: No sequence crosses split boundaries

**Implementation**:
1. Split timeline FIRST (before creating sequences)
2. Generate sequences WITHIN each split
3. A training sequence uses ONLY training data
4. A validation sequence uses ONLY validation data
5. A test sequence uses ONLY test data

**NEVER**:
- ❌ Create all sequences first, then split
- ❌ Allow sequences to span across boundaries

## Configuration

**Sequence Parameters**:
- Length: 50 timesteps (1 second @ 50 Hz)
- Stride: 1 (maximum overlap for training data)
- Rationale: Captures ~1 engine cycle, provides dense training data

**Split Ratios**:
- Train: 70% (0.02s to 506.46s)
- Validation: 15% (506.48s to 614.98s)
- Test: 15% (615.00s to 723.54s)

**Scaler**: StandardScaler (zero mean, unit variance)

## Execution Results

**Input**: 36,178 Phase 2 feature rows

**After warm-up removal**: 36,177 rows

**Chronological Split**:
- Train: 25,323 rows
- Val: 5,426 rows
- Test: 5,428 rows

**Generated Sequences**:
- Train: 25,274 sequences (50, 24)
- Val: 5,377 sequences (50, 24)
- Test: 5,379 sequences (50, 24)

**Total sequences**: 36,030

## Outputs

### 1. Sequence Arrays
**Location**: `data/sequences/`

- `X_train.npy`: (25274, 50, 24) - Training sequences
- `X_val.npy`: (5377, 50, 24) - Validation sequences
- `X_test.npy`: (5379, 50, 24) - Test sequences

**Format**: NumPy arrays (n_sequences, timesteps, features)

### 2. Fitted Scaler
**Location**: `models/feature_scaler.pkl`

**Type**: sklearn StandardScaler

**Fitted on**: Training data ONLY (25,323 samples)

**Purpose**: Ensures Phase 4/inference uses identical normalization

### 3. Metadata
**Location**: `data/sequences/metadata.json`

Contains:
- Input file path
- Feature names (24 features)
- Split configuration and actual row counts
- Time ranges for each split
- Sequence counts
- Scaler type and fitting confirmation
- Warm-up handling details

## Validation

Phase 3 validation performs 13 checks:

1. ✅ Output files exist
2. ✅ No NaN/Inf in sequences
3. ✅ Timeline chronological
4. ✅ Train/val/test chronological (no overlap)
5. ✅ No sequence crosses boundaries
6. ✅ Scaler is StandardScaler
7. ✅ Scaler fitted on training only
8. ✅ Same scaler for all splits
9. ✅ Sequence length = 50
10. ✅ Feature dimension = 24
11. ✅ No synthetic samples
12. ✅ Sequence counts consistent
13. ✅ Warm-up handled correctly

**Result**: ALL 13 CHECKS PASSED ✅

## How to Run

### Standalone Phase 3
```bash
cd integrated_pipeline
python phase3_sequence_preparation/prepare_sequences.py "data/features/extracted_features.csv"
```

### Integrated Pipeline (Phases 1+2+3)
```bash
cd integrated_pipeline
python run_pipeline.py --mode sequences --input "data/raw/my_telemetry.csv"
```

### Validation
```bash
cd integrated_pipeline
python phase3_sequence_preparation/validate.py --input "data/features/extracted_features.csv"
```

## Temporal Safety Verification

### Split Boundaries
```
Train:  [0.02s ────────────────────────── 506.46s]
                                              ↓ GAP
Val:                              [506.48s ─── 614.98s]
                                                  ↓ GAP  
Test:                                      [615.00s ── 723.54s]
```

**Verified**: No temporal overlap ✅

### Sequence Generation
```
Train sequences: Generated from [0.02s to 506.46s] ONLY
Val sequences:   Generated from [506.48s to 614.98s] ONLY
Test sequences:  Generated from [615.00s to 723.54s] ONLY
```

**Verified**: No boundary crossings ✅

### Scaler Fitting
```
1. Split timeline → [Train | Val | Test]
2. Fit scaler on Train ONLY
3. Transform Train with fitted scaler
4. Transform Val with SAME scaler
5. Transform Test with SAME scaler
```

**Verified**: Scaler fitted on training only ✅

## Warm-up Handling

**Issue**: First row (index 0) has NaN in std/rate features from Phase 2

**Solution**: Remove first row before split

**Impact**:
- Input: 36,178 rows
- After warm-up removal: 36,177 rows
- Loss: 1 row (0.003%)

**Verification**: No unexpected NaN remains in sequences ✅

## Why This Matters

### Data Leakage Prevention
If scaler were fitted on entire dataset:
- Training would "see" validation/test statistics
- Model would have unfair advantage
- Validation metrics would be optimistically biased
- Test performance would not generalize

**Our approach**: Scaler sees ONLY training data ✅

### Temporal Leakage Prevention
If sequences were created before splitting:
- Neighboring sequences share many samples
- Random split would put overlapping sequences in different sets
- Training could contain near-duplicates of validation/test
- Model would memorize rather than learn patterns

**Our approach**: Split FIRST, then generate sequences ✅

## Design Decisions

### Why 50 Timesteps?
- 1 second at 50 Hz sampling
- Captures ~1 engine cycle (e.g., 60 Hz at 3600 RPM)
- Balances context vs. computation
- Standard for engine telemetry

### Why Stride = 1?
- Maximum training data density
- Overlapping sequences provide more examples
- Sequences within same split can overlap safely
- Important: Overlap ONLY within splits, not across

### Why 70/15/15 Split?
- 70% training: Sufficient data for learning
- 15% validation: Model selection and early stopping
- 15% test: Final performance evaluation
- Standard practice for time-series

### Why StandardScaler?
- Zero mean, unit variance normalization
- Prevents features with large ranges from dominating
- Required for neural network training
- Reversible (can denormalize outputs)

## What Phase 3 Does NOT Do

❌ **Model Training**: Happens in Phase 4
❌ **Hyperparameter Tuning**: Happens in Phase 4
❌ **Reconstruction**: Happens in Phase 4
❌ **Anomaly Detection**: Happens in Phase 5
❌ **Threshold Determination**: Happens in Phase 5

Phase 3 ONLY prepares sequences. It does not build or train models.

## Next Steps (Phase 4 - NOT YET IMPLEMENTED)

After Phase 3 approval:
1. Build LSTM autoencoder architecture
2. Train on training sequences
3. Validate on validation sequences
4. Evaluate on test sequences
5. Calculate reconstruction errors

**DO NOT proceed to Phase 4 until Phase 3 is approved.**

## Technical Notes

- Implementation: `phase3_sequence_preparation/prepare_sequences.py`
- Validation: `phase3_sequence_preparation/validate.py`
- Uses: sklearn, numpy, pandas
- Sequence format: NumPy arrays (memory efficient, fast loading)
- Scaler format: Pickle (standard sklearn serialization)

## Troubleshooting

If validation fails:

1. **Check timeline**: Ensure no split overlap
2. **Check scaler**: Confirm fitted on training only
3. **Check sequences**: Verify no NaN/Inf
4. **Check boundaries**: Ensure sequences don't cross splits
5. **Check metadata**: Verify all counts match

## Questions or Issues?

If sequences look unexpected:
1. Check `metadata.json` for split details
2. Run validation: `python phase3_sequence_preparation/validate.py --input <csv>`
3. Inspect split timestamps
4. Verify scaler was saved correctly
5. Load sequences and check shapes/values
