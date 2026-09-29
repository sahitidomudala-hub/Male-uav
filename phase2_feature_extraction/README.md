> **Note:** channel lists, feature counts and parameter values in this file describe the original 10-channel prototype.
> The authoritative values are `channel_config.py`, `pipeline_config.py` and the top-level `README.md`.

# Phase 2: Feature Extraction

## Purpose

Phase 2 extracts time-domain features from 50 Hz engine telemetry data for downstream anomaly detection. These features capture:
- Current engine state (raw values)
- Recent behavior (moving statistics)
- Transients (rate of change)
- Deviations from baseline

## Input

**Source**: Phase 1 cleaned telemetry (`data/processed/cleaned_telemetry.csv`)

**Columns**:
- Time, EngTrq, RPM, MAT, MAP, EGT, FuelFlow, CH, BaroPress, TurboSpd, TPS

**Characteristics**:
- 50 Hz sampling rate
- Slow-varying engine telemetry (NOT high-frequency vibration)
- FuelFlow and TPS already corrected by Phase 1

## Feature Design

### What This Dataset IS
- 50 Hz sampled engine telemetry
- Captures thermodynamic and mechanical states
- Slow-varying signals (seconds to minutes timescale)

### What This Dataset IS NOT
- NOT 20 kHz vibration data
- NOT suitable for FFT-based frequency analysis
- Vibration features (crest factor, kurtosis, spectral analysis) belong to a DIFFERENT dataset

### Selected Features (25 Total)

#### 1. Raw Values (9 features)
Direct telemetry values capturing instantaneous state:
- `EngTrq_raw`, `RPM_raw`, `MAT_raw`, `MAP_raw`, `EGT_raw`
- `FuelFlow_raw`, `CH_raw`, `BaroPress_raw`, `TurboSpd_raw`, `TPS_raw`

**Why**: Capture current engine operating point

#### 2. Moving Statistics (10 features)
Trailing 1-second windows (50 samples @ 50Hz):
- `EngTrq_ma`, `EngTrq_std` - Torque trend and variability
- `RPM_ma`, `RPM_std` - Speed trend and variability
- `MAP_ma`, `MAP_std` - Pressure trend and variability
- `EGT_ma`, `EGT_std` - Exhaust temp trend and variability
- `FuelFlow_ma`, `FuelFlow_std` - Fuel consumption trend and variability

**Why**: Smooth noise, detect shifts, capture recent behavior

#### 3. Rate of Change (4 features)
Backward difference (current - previous):
- `EngTrq_rate` - Torque transient rate
- `RPM_rate` - Acceleration/deceleration
- `EGT_rate` - Thermal transient rate
- `TPS_rate` - Throttle movement rate

**Why**: Detect rapid transients and operational changes

#### 4. Time (1 feature)
- `Time` - Direct copy for temporal alignment

**Why**: Preserve timestamp synchronization

### Why These Features?

**Based on telemetry characteristics**:
- Features appropriate for 50 Hz sampling
- Capture slow dynamics (seconds to minutes)
- Suitable for engine state and transient detection

**From project documentation**:
- Slow telemetry → moving statistics, trends, rates
- NOT vibration → NO crest factor, kurtosis, FFT features

**Avoid unavailable variables**:
- NO AFR (not available)
- NO intake mass flow (not available)
- NO torque tracking error (no setpoint)
- NO model residuals (no physics model in Phase 2)

## Features NOT Included (And Why)

### Vibration Analysis Features (Wrong Dataset)
❌ **Crest Factor**: Designed for high-freq vibration impulsiveness  
❌ **Kurtosis**: Measures vibration spikes, not meaningful at 50 Hz  
❌ **Dominant Frequency**: FFT-based, for 20 kHz vibration dataset  
❌ **Spectral Centroid/Spread**: Frequency-domain features for vibration

### Unavailable/Not Derivable
❌ **AFR**: Not in dataset  
❌ **Intake Mass Flow**: Not available  
❌ **Torque Tracking Error**: Requires setpoint (not available)  
❌ **Model Residuals**: Requires physics model (Phase 2 scope)  
❌ **Compressor Pressure Ratio**: Needs additional sensors

## Temporal Leakage Prevention

**All features are CAUSAL** (no future-looking):

✅ **Moving windows**: Use past data only (trailing windows)
- pandas `rolling()` with default behavior (backwards-looking)
- Window at time `t` uses: `[t-49, t-48, ..., t-1, t]`

✅ **Rate calculations**: Backward difference only
- `rate[t] = value[t] - value[t-1]`

❌ **NO centered windows** (would use future data)  
❌ **NO forward-looking calculations**

### Example at Time t=100:
- Moving average uses samples [51, 52, ..., 99, 100] ✓
- Rate uses: value[100] - value[99] ✓
- Does NOT use samples [101, 102, ...] ✓

## Window Selection

**Window size**: 50 samples = 1.0 second @ 50 Hz

**Rationale**:
- Short enough to capture transients
- Long enough to smooth sensor noise
- Matches ~1 engine cycle at typical speeds (e.g., 60 Hz at 3600 RPM)
- Standard practice for engine telemetry

## NaN Handling

**Warm-up period**: First sample has NaN for std and rate features

**Why**:
- `_std` features: Need 2 samples minimum for std deviation
- `_rate` features: Need previous sample for difference

**Strategy**:
- NaNs documented and expected
- Rows NOT dropped (preserve alignment)
- Phase 3 will handle edge effects during sequence preparation

**Impact**: 1 NaN out of 36,178 samples (0.003%) - negligible

## Quality Flags Integration

Phase 1 generated quality flags for corrected FuelFlow and TPS samples.

**Approach**:
- Features calculated from CLEANED values (Phase 1 output)
- Quality flags remain in `quality_flags.csv` for traceability
- No rows discarded
- Downstream phases can filter/weight based on quality if needed

## Outputs

### 1. `data/features/extracted_features.csv`
- 36,178 rows × 25 columns
- Same row count as Phase 1 input (alignment preserved)
- Features ready for Phase 3 sequence preparation

### 2. `phase2_feature_extraction/feature_definitions.json`
Complete documentation for every feature:
```json
{
  "feature_name": "EngTrq_ma",
  "source_columns": ["EngTrq"],
  "calculation": "Trailing mean over 50 samples",
  "window_size": 50,
  "units": "Nm",
  "causal": true,
  "justification": "Recent average EngTrq (smoothed)"
}
```

### 3. `data/features/phase2_validation.json`
Validation results (row counts, NaN checks, alignment verification)

## How to Run

### Standalone Phase 2
```bash
cd integrated_pipeline
python phase2_feature_extraction/extract_features.py "data/processed/cleaned_telemetry.csv"
```

### Integrated Pipeline (Phase 1 + Phase 2)
```bash
cd integrated_pipeline
python run_pipeline.py --mode features --input "data/raw/my_telemetry.csv"
```

### Validation
```bash
cd integrated_pipeline
python phase2_feature_extraction/validate.py --input "data/processed/cleaned_telemetry.csv"
```

## Validation Checks

Phase 2 validation verifies:

1. ✅ Feature output exists
2. ✅ Row count maintained (no deletions)
3. ✅ Time remains ordered
4. ✅ Time aligned with Phase 1 input
5. ✅ All features are causal (no future-looking)
6. ✅ Feature definitions match implementation
7. ✅ Only available columns used (no invented variables)
8. ✅ NaN values only in expected warm-up
9. ✅ No Inf values
10. ✅ Phase 1 output unchanged

## Results (Current Dataset)

**Input**: Teammate_Telemetry_Data 3.csv (via Phase 1)
- 36,178 samples @ 50 Hz
- 11 telemetry channels
- Duration: 723.54 seconds (~12 minutes)

**Output**:
- 36,178 rows × 25 features
- All validations: ✅ PASSED
- NaN counts: 1 per std/rate feature (expected)
- Inf values: 0

**Feature Categories**:
- Raw values: 10 features
- Moving statistics: 10 features
- Rate of change: 4 features
- Time: 1 feature

## Design Decisions

### Why NOT Use Vibration Features?

The existing `Phase_2_Feature_Extraction` implementation was designed for **20 kHz vibration data**. It included:
- RMS, crest factor, kurtosis (vibration impulsiveness metrics)
- FFT dominant frequency, spectral centroid (frequency analysis)
- 100ms windows on high-frequency signals

**These are NOT applicable to 50 Hz engine telemetry** because:
1. **Wrong frequency regime**: 50 Hz vs. 20 kHz (400x difference)
2. **Wrong signal type**: Slow thermodynamic states vs. mechanical vibration
3. **Wrong physics**: Engine operating point vs. bearing/structural vibration

### Why These Features Instead?

50 Hz telemetry captures **slow engine dynamics**:
- Torque, speed, temperatures, pressures
- Changes over seconds to minutes
- Suitable for: moving averages, std deviations, rates of change

This matches the **documented slow-telemetry feature concepts**.

## What Phase 2 Does NOT Do

❌ **Normalization/Scaling**: Happens in Phase 3 (train-only scaler)  
❌ **Train/Val/Test Split**: Happens in Phase 3 (chronological)  
❌ **Sequence Creation**: Happens in Phase 3 (overlapping windows)  
❌ **Feature Selection**: All features retained for now  
❌ **Dimensionality Reduction**: Not in scope for Phase 2

## Next Steps (Phase 3 - NOT YET IMPLEMENTED)

After Phase 2 approval:
1. **Chronological split**: Train/validation/test (avoid temporal leakage)
2. **Sequence preparation**: Overlapping windows for LSTM
3. **Normalization**: Fit scaler on TRAINING data only (avoid data leakage)
4. **Edge handling**: Manage NaNs from feature warm-up

**DO NOT proceed to Phase 3 until Phase 2 is approved.**

## Technical Notes

- Implementation: `phase2_feature_extraction/extract_features.py`
- Validation: `phase2_feature_extraction/validate.py`
- Feature design: `phase2_feature_extraction/FEATURE_DESIGN.md`
- Uses pandas for vectorized operations (efficient)
- No external ML libraries needed for Phase 2

## Questions or Issues?

If features look unexpected:
1. Check `feature_definitions.json` for calculation details
2. Run validation: `python phase2_feature_extraction/validate.py --input <csv>`
3. Compare with `FEATURE_DESIGN.md` for rationale
4. Verify Phase 1 cleaned telemetry is correct input
