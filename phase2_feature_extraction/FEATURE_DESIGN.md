> **Note:** channel lists, feature counts and parameter values in this file describe the original 10-channel prototype.
> The authoritative values are `channel_config.py`, `pipeline_config.py` and the top-level `README.md`.

# Phase 2 Feature Extraction - Design Document

## Dataset Context

**Input**: Cleaned 50 Hz engine telemetry from Phase 1
**Type**: Slow telemetry (NOT high-frequency vibration)
**Channels**: EngTrq, RPM, MAT, MAP, EGT, FuelFlow, CH, BaroPress, TurboSpd, TPS

## Feature Design Philosophy

### What This Dataset IS:
- 50 Hz sampled engine telemetry
- Captures thermodynamic and mechanical states
- Slow-varying signals (seconds to minutes timescale)
- Suitable for: current values, trends, rates of change, deviations

### What This Dataset IS NOT:
- NOT 20 kHz vibration data
- NOT suitable for: FFT dominant frequency, crest factor, kurtosis, spectral analysis
- These vibration features belong to a DIFFERENT dataset

## Feature Categories for 50 Hz Telemetry

### 1. Current/Raw Values
**Why**: Capture instantaneous state
**Features**: Direct telemetry values at each timestamp

### 2. Moving Statistics (Trailing Windows)
**Why**: Capture recent behavior, smooth noise, detect shifts
**Features**:
- Moving mean (trailing window)
- Moving std deviation (trailing window)

### 3. Rate of Change
**Why**: Detect rapid transients, acceleration/deceleration
**Features**:
- First difference (rate of change)
- Can be calculated per-sample or over short windows

### 4. Deviation from Recent Baseline
**Why**: Detect anomalies as deviations from recent normal
**Features**:
- Deviation from moving mean
- Normalized deviation (z-score style)

## Selected Features for Implementation

Based on slow telemetry characteristics and avoiding unavailable variables:

| Feature Name | Source Channel(s) | Calculation | Window | Causal | Justification |
|--------------|-------------------|-------------|---------|--------|---------------|
| Time | Time | Direct copy | N/A | Yes | Timestamp alignment |
| EngTrq_raw | EngTrq | Direct value | N/A | Yes | Current torque |
| EngTrq_ma | EngTrq | Trailing mean | 50 samples (1s @ 50Hz) | Yes | Recent avg torque |
| EngTrq_std | EngTrq | Trailing std | 50 samples | Yes | Torque variability |
| EngTrq_rate | EngTrq | diff() | 1 sample | Yes | Torque rate of change |
| RPM_raw | RPM | Direct value | N/A | Yes | Current speed |
| RPM_ma | RPM | Trailing mean | 50 samples | Yes | Recent avg speed |
| RPM_std | RPM | Trailing std | 50 samples | Yes | Speed variability |
| RPM_rate | RPM | diff() | 1 sample | Yes | Acceleration/deceleration |
| MAP_raw | MAP | Direct value | N/A | Yes | Current manifold pressure |
| MAP_ma | MAP | Trailing mean | 50 samples | Yes | Recent avg pressure |
| MAP_std | MAP | Trailing std | 50 samples | Yes | Pressure variability |
| EGT_raw | EGT | Direct value | N/A | Yes | Current exhaust temp |
| EGT_ma | EGT | Trailing mean | 50 samples | Yes | Recent avg exhaust temp |
| EGT_std | EGT | Trailing std | 50 samples | Yes | Exhaust temp variability |
| EGT_rate | EGT | diff() | 1 sample | Yes | Thermal transient rate |
| FuelFlow_raw | FuelFlow | Direct value | N/A | Yes | Current fuel flow |
| FuelFlow_ma | FuelFlow | Trailing mean | 50 samples | Yes | Recent avg fuel flow |
| FuelFlow_std | FuelFlow | Trailing std | 50 samples | Yes | Fuel flow variability |
| TPS_raw | TPS | Direct value | N/A | Yes | Current throttle position |
| TPS_rate | TPS | diff() | 1 sample | Yes | Throttle movement rate |
| MAT_raw | MAT | Direct value | N/A | Yes | Current intake temp |
| TurboSpd_raw | TurboSpd | Direct value | N/A | Yes | Current turbo speed |
| CH_raw | CH | Direct value | N/A | Yes | Cylinder head temp |
| BaroPress_raw | BaroPress | Direct value | N/A | Yes | Barometric pressure |

**Total Features**: 25

## Features NOT Included (And Why)

### From Vibration Analysis (Not Applicable):
- ❌ **Crest Factor**: Requires high-freq vibration (peak/RMS ratio for impulsiveness)
- ❌ **Kurtosis**: Measures impulsiveness in vibration, not meaningful at 50 Hz telemetry
- ❌ **Dominant Frequency**: FFT-based, designed for 20 kHz vibration analysis
- ❌ **Spectral Centroid/Spread**: Frequency-domain features for vibration

### Unavailable/Not Derivable:
- ❌ **AFR (Air-Fuel Ratio)**: Not available in current dataset
- ❌ **Intake Mass Flow**: Not available
- ❌ **Torque Tracking Error**: Requires setpoint (not available)
- ❌ **Model Residuals**: Requires physics model (not in scope for Phase 2)
- ❌ **Compressor Pressure Ratio**: Would need additional sensors
- ❌ **Thermal Efficiency**: Requires additional calculations not justified yet

## Temporal Leakage Prevention

**All features are CAUSAL (trailing)**:
- Moving windows use **past data only** (pandas rolling with `min_periods`)
- Rate calculations use **backward difference** (current - previous)
- NO centered windows
- NO future-looking calculations

**Example at time t**:
- Moving average uses: [t-49, t-48, ..., t-1, t]
- Rate uses: value[t] - value[t-1]

## Window Selection

**Window size**: 50 samples = 1 second @ 50 Hz

**Rationale**:
- Short enough to capture transients
- Long enough to smooth sensor noise
- Matches approx 1 engine cycle at typical operating speeds
- Standard practice for engine telemetry

## NaN Handling

**First 49 samples** of moving features will have NaN due to insufficient history.

**Strategy**:
- Document this warm-up period
- Do NOT drop rows
- Retain Time alignment
- Phase 3 sequence preparation will handle edge effects

## Quality Flags Integration

Phase 1 generated quality flags for FuelFlow and TPS corrections.

**Approach**:
- Features are calculated from CLEANED values (Phase 1 output)
- Quality flags remain available in separate CSV for traceability
- No rows discarded due to corrections
- Downstream phases can weight/filter based on quality if needed

## Implementation Notes

1. Use pandas rolling() with min_periods for moving features
2. Use pandas diff() for rate features
3. Preserve exact timestamp/row alignment with Phase 1 output
4. No resampling, no aggregation, no row deletion
5. Output same number of rows as input (36,178 for current dataset)

## Next Steps (After Phase 2)

- Phase 3: Chronological split + sequence preparation
- Normalization will happen in Phase 3 (train-only scaler)
- Phase 2 outputs RAW feature values (no scaling yet)
