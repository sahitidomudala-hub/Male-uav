# Edge AI Layer -- Engine Anomaly Detection (Layer 2)

Backend layer of the UAV engine Digital Twin / IVHM system:
`DAQ -> **Edge AI (this repo)** -> Secure Telemetry -> Digital Twin Core (PINN) -> Macro-Analytics`.

Responsibilities of this layer: **real-time signal denoising, instantaneous anomaly detection,
autonomous fault isolation** (which channel deviates most). No UI. RUL, physics-based state
estimation and predictive maintenance belong to other layers / are deferred.

## Design: nothing is hardcoded in logic files

| Where | What lives there |
|---|---|
| `channel_config.py` | one row per channel: correction rule, moving-stats / rate features, filter, `model_input`, unit |
| `pipeline_config.py` | every global parameter: split, sequence length, thresholds, filter-derivation settings, model/training settings, severity tiers, sweep settings, fault presets |
| trained artifacts | everything tied to the model: feature names and order, sequence length, stride, sampling rate, warm-up rows, redundant channel groups (`data/sequences/metadata.json`), scaler, filter cutoffs (`models/filter_params.json`), threshold (`outputs/anomaly_detection_report.json`) |
| `schema_guard.py` | refuses to run if the config no longer matches the trained artifacts ("retrain required") |

**Add / change a channel:** edit only `channel_config.py`, re-run Phases 1-3, retrain (the model input
size changes). The guard turns a forgotten retrain into an error instead of silent misuse.

## Run (PowerShell, from the repo root, using the project venv)

```powershell
$env:KERAS_BACKEND="tensorflow"
$py = ".venv\Scripts\python.exe"

# 1. Prepare data (fits filter cutoffs + scaler on the TRAINING rows only)
& $py phase1_preprocessing/preprocess.py data/raw/Simulink_Telemetry_Run.csv
& $py phase2_feature_extraction/extract_features.py data/processed/cleaned_telemetry.csv
& $py phase3_sequence_preparation/prepare_sequences.py data/features/extracted_features.csv data/sequences

# 2. Train (slow; run uninterrupted)
& $py phase4_autoencoder/train_autoencoder.py --sequences-dir data/sequences --output-dir .

# 3. Threshold, attribution, validation evidence, alerts
& $py phase5_anomaly_detection/detect_anomalies.py --base-dir .
& $py phase5_anomaly_detection/attribute_anomalies.py --base-dir .
& $py phase5_anomaly_detection/bootstrap_threshold.py --base-dir .
& $py phase5_anomaly_detection/fault_injection_validation.py --base-dir .
& $py phase5_anomaly_detection/noise_sweep.py --base-dir .
& $py phase5_anomaly_detection/generate_alerts.py --base-dir .
& $py phase5_anomaly_detection/trend_analysis.py --base-dir .

# 4. Validate every phase
& $py phase1_preprocessing/validate.py data/raw/Simulink_Telemetry_Run.csv data/processed
& $py phase2_feature_extraction/validate.py --input data/processed/cleaned_telemetry.csv
& $py phase3_sequence_preparation/validate.py --input data/features/extracted_features.csv
& $py phase4_autoencoder/validate.py --base-dir .
& $py phase5_anomaly_detection/validate.py --base-dir .

# 5. Score a NEW file with the saved artifacts (nothing refit / retrained / recalculated)
& $py run_pipeline.py --mode inference --input path\to\new_telemetry.csv
```
Scripts locate the repo by their own path, so they also work from another working directory.

## Pipeline

1. **Phase 1** validate schema/timestamps -> estimate per-channel noise (report) -> causal filter ->
   correction rule. Butterworth cutoffs marked `'auto'` are derived from each channel's bandwidth on the
   training rows, saved, and **loaded (never re-derived)** at inference. Filters start at steady state so
   large DC offsets (K, Pa) create no start-up transient. All filters are causal.
2. **Phase 2** causal features for `model_input` channels (raw, trailing mean/std, backward difference);
   feature count is computed from the config and checked.
3. **Phase 3** chronological 70/15/15 split, sequences within each split, scaler fitted on train only,
   redundant channel groups found from train rows, metadata written (incl. config fingerprint).
4. **Phase 4** LSTM autoencoder (parameters from `pipeline_config.py`).
5. **Phase 5** threshold = percentile of VALIDATION errors; per-channel attribution; bootstrap CI;
   fault-injection and noise-sweep validation; alert grouping; trend description.

`Altitude` and `BaroPress` are `model_input=False`: altitude climbs monotonically, so under a chronological
split all validation/test samples lie outside the training range. They stay in the handoff as context.

## Verified Results (40-feature model, trained on `Simulink_Telemetry_Run.csv`)

Architecture: `Input(50,40) -> LSTM(64) -> RepeatVector(50) -> LSTM(64) -> TimeDistributed(Dense(40))`,
62,504 trainable parameters. Training converged via early stopping at 83/100 epochs
(final train loss 0.00141, final val loss 0.00618).

- **Threshold:** 0.024915 (95th percentile of validation reconstruction error).
- **Test-set anomaly rate:** 15.39% (1,230/7,994 sequences), grouped into **10 distinct events**
  (4 HIGH, 4 MEDIUM, 2 LOW severity; 4 flagged for immediate uplink) after merging overlapping
  windows -- see `outputs/onboard_alert_log.jsonl`. All 10 events fall in the last ~15% of the
  mission timeline (t = 924.9-1050.8 s) and are dominated by TPS (5 events) and EngTrq (4 events).
- **Threshold stability:** bootstrap 95% CI [0.0241, 0.0261] -> anomaly rate stable between
  14.76%-16.61% across resampling (`outputs/threshold_bootstrap_ci.json`). This is a well-calibrated
  threshold, not a fragile point estimate.
- **Fault-injection validation:** 95% detection (138/145 valid scenarios,
  `outputs/fault_injection_validation.csv`; 49 scenarios x 3 locations in the test split, 2 vacuous
  perturbations excluded). Reliable on abrupt step/scale/set changes (100% on `set`, `scale`,
  `ramp_scale`, `ramp_to`); known weaker detection on gradual ramps and frozen-sensor patterns
  (notably a simulated TPS drift and an RPM sensor freeze were not reliably caught).
- **Attribution accuracy (among detected scenarios):** top-1 channel matched the perturbed channel
  110/138 times; the true channel appeared in the top-3 for 121/138.
- **Noise sensitivity** (`outputs/noise_sweep.csv`): RPM, EngTrq, EGT, FuelFlow, TPS, OilPress and
  BusVoltage start showing elevated false-flag rates at noise as low as 0.05-0.25 standard deviations
  (channel-specific, measured against each channel's own training-set variability); MAT, CH, TurboSpd,
  OilTemp, InjTiming and the vibration channels tolerate noise up to ~2 standard deviations before
  flagging. This was measured with synthetic white Gaussian noise on a noise-free simulation -- it has
  **not** been validated against real sensor noise, and is expected to change once real DAQ noise
  characteristics are known.

**Do not confuse this 15.39% with the number reported by `--mode inference` when run on the full
training file (~3.7% in our own testing).** That run scores the entire file, including the ~85% of
rows the model trained/validated on, which dilutes the rate. The 15.39% figure above, computed by
`detect_anomalies.py` on the held-out test split only, is the number that reflects generalization and
is the one to cite.

## Outputs and what they mean

| File | Meaning |
|---|---|
| `outputs/inference_results.csv` | per sequence: error, threshold, Normal/Anomaly |
| `outputs/downstream_health_vector.csv` (+ `_units.json`) | candidate handoff, see `DOWNSTREAM_HANDOFF.md` |
| `outputs/inference_noise_estimate.json` | per-channel high-frequency noise estimate of the scored file |
| `outputs/anomaly_attribution.csv` | top channels per sequence (deviation ranking, not fault isolation) |
| `outputs/threshold_bootstrap_ci.json` | how stable the threshold is |
| `outputs/fault_injection_validation.csv` | labelled synthetic faults: detected? attributed? (raw-domain injection through the real pipeline, several locations, auto sweep over every channel) |
| `outputs/noise_sweep.csv` | how much added noise is tolerated, with and without the Phase 1 filters |
| `outputs/onboard_alert_log.jsonl`, `priority_alerts.jsonl` | events (consecutive anomalous windows merged) with severity; HIGH flagged for immediate uplink |
| `outputs/trend_analysis.json` | descriptive error-over-time summary |

Inference scratch files go to `data/inference/`, so scoring a new file never overwrites the training-side
`data/processed` / `data/features` that the Phase 5 analysis scripts use.

## Known limitations (state these plainly)

- The current DAQ data is a **fault-free, noise-free Simulink simulation**. The anomaly rate on it is set by
  the percentile threshold (a share of healthy validation data is flagged by construction) -- it is not a fault
  rate, and no accuracy/precision/recall can be claimed. Evidence of detection ability = the synthetic
  fault-injection and noise-sweep results above.
- Vibration values are engineering estimates, not measured data. `Vib2X` and `InjTiming` are exact functions
  of `Vib0p5X` and `FuelFlow` in this simulation (detected automatically as redundant groups), so attribution
  across them is shared, not independent.
- Attribution ranks which channel's pattern deviates most from the learned normal; it is not fault isolation
  or root cause.
- Filter cutoffs derived from noise-free training data are bandwidth-based, not noise-optimised; how much real
  sensor noise is tolerated must be read from `noise_sweep.csv`. The model is trained on noise-free data, and
  its noise sensitivity above has not yet been checked against a real noisy sample.
- Timestamps are mission-elapsed seconds, not UTC.
- Live streaming ingestion is not implemented; inference works on files.
- Onboard performance (latency, memory on embedded hardware) is untested.
- RUL, physics-based degradation states and predictive maintenance are not part of this layer.
