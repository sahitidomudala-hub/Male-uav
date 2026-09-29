# Downstream Health Vector Handoff

`outputs/downstream_health_vector.csv` gives sequence-level anomaly information plus a representative value
of every telemetry channel for downstream layers (Secure Telemetry, PINN, ...).

**IMPORTANT**: this is a candidate schema. The project documents do not mandate one; the exact interface must
be agreed with the downstream teammates. Units are in `outputs/downstream_health_vector_units.json`
(generated from `channel_config.py`).

## Current calibration (for reference, not guaranteed to be final)

Threshold: 0.024915. Test-split anomaly rate: 15.39% (1,230/7,994 sequences), grouped into 10 distinct
events (4 HIGH, 4 MEDIUM, 2 LOW severity), all within the last ~15% of the mission timeline and dominated
by TPS/EngTrq deviations. Fault-injection validation: 95% detection (138/145 scenarios). See the top-level
`README.md` "Verified Results" section for the full evidence (bootstrap CI, fault-injection and noise-sweep
results) backing this calibration. This model has not been validated against real sensor noise.

## Columns

| Column | Meaning |
|---|---|
| `sequence_index` | 0-based index of the scored window in this run |
| `start_time`, `end_time` | first / last sample time of the window, mission-elapsed seconds (not UTC) |
| `reconstruction_error` | mean squared reconstruction error of the window |
| `threshold` | the saved (loaded, not recalculated) threshold |
| `classification` | `Normal` / `Anomaly` (`error > threshold`) |
| `<Channel>_mean` (all configured channels) | MEAN of the cleaned (filtered + corrected) channel over the window. It is a compact summary, NOT the official PINN input |

Channels that are passthrough-only for the model (`model_input=False`, currently Altitude and BaroPress) are
included as context.

## Sequence-to-time mapping (read from the training metadata, not hardcoded)

For sequence `i`: cleaned-telemetry rows `[w + i*stride, w + i*stride + L - 1]`, where `L` = sequence length,
`stride` = stride and `w` = number of warm-up rows dropped in Phase 2/3 (all recorded in
`data/sequences/metadata.json`). `end_time` is the time of the LAST sample of the window.

## For the PINN teammate

If per-sample telemetry is needed, use the cleaned stream (`data/processed/cleaned_telemetry.csv` from training,
or `data/inference/processed/cleaned_telemetry.csv` after scoring a new file), not the window means. Open questions
to settle: per-sample vs window values, filtered vs raw, whether anomaly flags are an input or a reference signal,
timestamp convention.

## Anomaly Information

### reconstruction_error
The mean squared error (MSE) between the original sequence and the LSTM autoencoder's reconstruction. Higher values indicate the sequence deviates from learned normal patterns.

### threshold  
The anomaly threshold: the configured percentile (`THRESHOLD_PERCENTILE`) of reconstruction errors on the validation set only, saved in `outputs/anomaly_detection_report.json` and loaded (never recalculated) at inference. Currently 0.024915 -- see "Current calibration" above.

### classification
- **"Normal"**: reconstruction_error ≤ threshold
- **"Anomaly"**: reconstruction_error > threshold

## Important Limitations

1. **Sequence-level classification**: The anomaly classification applies to the entire sequence window, not individual samples.

2. **Attribution is a deviation ranking, not fault isolation**: the autoencoder gives a sequence-level error. `outputs/anomaly_attribution.csv` ranks which channels deviate most from their own normal range; it does not prove which sensor or component failed. Redundant channels (see `redundant_groups` in `data/sequences/metadata.json` -- currently `FuelFlow`/`InjTiming` and `Vib0p5X`/`Vib2X`) share attribution. Fault-injection testing shows this ranking is correct for the perturbed channel 110/138 times (top-1) and 121/138 times (top-3) among detected scenarios.

3. **No confirmed faults**: Without ground-truth fault labels, "Anomaly" sequences represent deviations from learned normal behavior that require further investigation, not confirmed engine faults. The current dataset is a fault-free simulation, so the 15.39% test-split rate reflects the calibrated threshold, not a real fault incidence.

4. **Inference mode only**: This handoff is generated during inference. The model, scaler, and threshold are loaded from existing trained artifacts and are NOT retrained or recalculated.

## Usage Example

```python
import pandas as pd

# Load handoff
handoff = pd.read_csv('outputs/downstream_health_vector.csv')

# Filter anomalous sequences
anomalies = handoff[handoff['classification'] == 'Anomaly']

# Get time range and mean telemetry for first anomaly
first_anomaly = anomalies.iloc[0]
print(f"Anomaly at t=[{first_anomaly['start_time']:.2f}, {first_anomaly['end_time']:.2f}]s")
print(f"Mean RPM: {first_anomaly['RPM_mean']:.1f}")
print(f"Reconstruction error: {first_anomaly['reconstruction_error']:.4f}")
```

## Integration Notes

This handoff format is designed to provide comprehensive information for downstream processing. The downstream PINN/Secure Telemetry teammate should:

1. Review the available fields
2. Identify which fields are actually needed for their interface
3. Specify the final agreed-upon schema

Not all fields may be required. The schema can be adjusted based on actual downstream requirements.

## Generation

The handoff is generated automatically during inference mode:

```bash
python run_pipeline.py --mode inference --input "path/to/telemetry.csv"
```

Both `outputs/inference_results.csv` (minimal 4-column format) and `outputs/downstream_health_vector.csv`
(26-column format: 6 fixed columns + one `_mean` column per configured channel, currently 20 channels) are
created.
