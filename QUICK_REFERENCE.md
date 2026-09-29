# Quick Reference -- Edge AI layer

**Score a new file** (after training): `python run_pipeline.py --mode inference --input <csv>`
**Retrain from scratch**: Phases 1-4 in `README.md` (`--mode train/detect` retrain the model).

| Need to change | Edit | Then |
|---|---|---|
| add/remove/alter a channel, filter, correction, unit | `channel_config.py` | rerun Phases 1-3, retrain |
| split, sequence length, model size, epochs, threshold percentile, severity tiers, sweep/fault settings | `pipeline_config.py` | retrain if it affects features/model (the schema guard tells you) |

Channel columns (Time + 20): EngTrq, RPM, MAT, MAP, EGT, FuelFlow, CH, BaroPress, TurboSpd, TPS, OilPress,
OilTemp, AltCurrent, BusVoltage, InjTiming, Altitude, Vib0p5X, Vib1X, Vib2X, VibRMS.
Units: `channel_config.py` (also written to `outputs/*_units.json`). Model features are computed from the config.

Key artifacts: `models/lstm_autoencoder.keras`, `models/feature_scaler.pkl`, `models/filter_params.json`,
`data/sequences/metadata.json`, `outputs/anomaly_detection_report.json` (threshold).

Errors of the form "RETRAIN REQUIRED" come from `schema_guard.py`: config and trained artifacts disagree.
Use the project venv (`.venv\Scripts\python.exe`) with `KERAS_BACKEND=tensorflow`.
