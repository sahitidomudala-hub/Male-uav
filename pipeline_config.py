"""
pipeline_config.py
==================
SINGLE SOURCE OF TRUTH for pipeline-wide parameters.

All numeric constants that control pipeline behavior live here. To change
sequence length, split ratios, thresholds, severity tiers, or any other
pipeline-wide setting, edit ONLY this file (and retrain if the change affects
the model's input dimension or training procedure).

DO NOT edit these values in individual scripts -- they should all read from
here.
"""

# =============================================================================
# PHASE 3: Sequence Preparation
# =============================================================================

SEQUENCE_LENGTH = 50  # timesteps per sequence (must match training)
STRIDE = 1  # sliding window stride
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # Must sum to 1.0

# Moving window size for Phase 2 features (seconds)
MOVING_WINDOW_SECONDS = 1.0  # 1-second trailing window for MA/std

# =============================================================================
# PHASE 4: LSTM Autoencoder Training
# =============================================================================

EPOCHS = 100
BATCH_SIZE = 32
INFERENCE_BATCH_SIZE = 64  # For prediction only (not training)
LEARNING_RATE = 0.001
LSTM_UNITS = 64  # Number of units in encoder/decoder LSTMs
OPTIMIZER = 'adam'
LOSS = 'mse'
METRICS = ['mae']

# =============================================================================
# PHASE 5: Anomaly Detection
# =============================================================================

THRESHOLD_PERCENTILE = 95.0  # Percentile of validation errors for threshold

# =============================================================================
# PHASE 5 EXTENSIONS: Severity, Bootstrap, Trend
# =============================================================================

# Severity tier definitions (ratio of error/threshold)
# Format: list of (upper_bound, label) tuples, ascending order
SEVERITY_THRESHOLDS = [
    (1.5, "LOW"),
    (3.0, "MEDIUM"),
    (float("inf"), "HIGH"),
]

# Which severity levels trigger immediate uplink (bandwidth-constrained telemetry)
IMMEDIATE_UPLINK_SEVERITIES = {"HIGH"}

# Bootstrap threshold stability analysis
BOOTSTRAP_N_RESAMPLES = 2000
BOOTSTRAP_CI_LEVEL = 95.0  # Confidence interval percentage

# Trend analysis
TREND_ROLLING_WINDOW = 50  # sequences for rolling average smoothing
TREND_R_SQUARED_THRESHOLD = 0.3  # R² threshold for "significant trend"
TREND_SLOPE_THRESHOLD = 1e-6  # Minimum |slope| to consider non-flat

# =============================================================================
# PHASE 5 EXTENSIONS: Fault Injection & Noise Sweep
# =============================================================================

# Auto-sweep: for every model_input channel, perturb by k * sigma_train
FAULT_INJECTION_SIGMA_MULTIPLES = [5.0]  # k values for step_offset and ramp_offset

# Named fault scenarios (name, operation, channels, value, expected_channels)
# Operations: 'step_offset', 'ramp_offset', 'scale', 'set', 'freeze'
# Named scenario presets are DATA, not code: name, op, channels, value,
# (optional) expected_channels. Ops: step_offset, ramp_offset (0 -> value),
# scale, ramp_scale (1 -> value), set, freeze, ramp_to (current -> value).
# Values are in the channel's own unit (see channel_config units). Expected
# channels are extended automatically with redundant-group members.
# Independent of these, an AUTO SWEEP perturbs every model_input channel by
# +k standard deviations (FAULT_INJECTION_SIGMA_MULTIPLES) with no channel
# names or magnitudes typed anywhere.
FAULT_INJECTION_SCENARIOS = [
    {'name': 'fuel_cutoff',           'op': 'set',         'channels': ['FuelFlow'],  'value': 0.0},
    {'name': 'sensor_freeze_rpm',     'op': 'freeze',      'channels': ['RPM'],       'value': None},
    {'name': 'torque_dropout',        'op': 'ramp_to',     'channels': ['EngTrq'],    'value': -20.0},
    {'name': 'egt_spike',             'op': 'step_offset', 'channels': ['EGT'],       'value': 300.0},
    {'name': 'map_dropout',           'op': 'set',         'channels': ['MAP'],       'value': 101325.0},
    {'name': 'vibration_rise',        'op': 'scale',       'channels': ['Vib0p5X', 'Vib1X', 'Vib2X', 'VibRMS'], 'value': 3.0},
    {'name': 'inj_timing_offset',     'op': 'step_offset', 'channels': ['InjTiming'], 'value': 10.0},
    {'name': 'oil_pressure_drop',     'op': 'scale',       'channels': ['OilPress'],  'value': 0.4},
    {'name': 'oil_pressure_gradual',  'op': 'ramp_scale',  'channels': ['OilPress'],  'value': 0.4},
    {'name': 'oil_temp_rise',         'op': 'step_offset', 'channels': ['OilTemp'],   'value': 25.0},
    {'name': 'bus_voltage_sag',       'op': 'step_offset', 'channels': ['BusVoltage'], 'value': -2.0},
    {'name': 'alternator_loss',       'op': 'set',         'channels': ['AltCurrent'], 'value': 0.0},
    {'name': 'ch_overheat',           'op': 'step_offset', 'channels': ['CH'],        'value': 40.0},
]
FAULT_INJECTION_LOOKBACK_SECONDS = 10.0      # real telemetry before the injected window (lets filters/rolling stats settle)
INJECTION_BASE_FRACTIONS_OF_TEST = [0.2, 0.5, 0.8]   # where in the test split base blocks are taken (0..1); the noise sweep uses the middle one
VACUOUS_PERTURBATION_MIN_SD = 1e-3           # a perturbation changing the signal by less than this many training-sigmas is reported as vacuous (e.g. freezing an already-constant signal) and excluded from detection rates
NOISE_SWEEP_LEVELS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
NOISE_SWEEP_BLOCK_LENGTH_SECONDS = 10.0  # Contiguous block within test split
NOISE_SWEEP_SEED = 42
NOISE_SWEEP_FLAGGED_MARGIN = 0.10  # Threshold: flagged must exceed baseline by this fraction

# =============================================================================
# REDUNDANCY DETECTION
# =============================================================================

# Channels with |Pearson r| >= this cutoff on raw values (train split only)
# are flagged as redundant
REDUNDANCY_CORRELATION_CUTOFF = 0.9999

# =============================================================================
# FEATURE SUFFIXES (Phase 2)
# =============================================================================

# Feature naming convention: {channel}{suffix}
# These are the ONLY suffixes used; no hardcoded "_raw", "_ma", etc. in logic
FEATURE_SUFFIX_RAW = '_raw'
FEATURE_SUFFIX_MA = '_ma'
FEATURE_SUFFIX_STD = '_std'
FEATURE_SUFFIX_RATE = '_rate'


def get_feature_suffixes():
    """Return all feature suffixes in canonical order."""
    return {
        'raw': FEATURE_SUFFIX_RAW,
        'ma': FEATURE_SUFFIX_MA,
        'std': FEATURE_SUFFIX_STD,
        'rate': FEATURE_SUFFIX_RATE,
    }


def compute_expected_feature_count(channel_config):
    """
    Compute the expected number of features given a channel configuration.
    
    Args:
        channel_config: dict from channel_config.CHANNEL_CONFIG
    
    Returns:
        int: Expected feature count
    """
    count = 0
    for channel, cfg in channel_config.items():
        if not cfg.get('model_input', True):  # Default to True for backward compat
            continue
        count += 1  # raw feature always present
        if cfg.get('moving_stats', False):
            count += 2  # ma + std
        if cfg.get('rate', False):
            count += 1  # rate
    return count


# ============================================================================
# FILTERING (Phase 1) -- cutoffs derived from the TRAINING data's own bandwidth
# ============================================================================
# A Butterworth channel declared as ('butterworth', 'auto') in channel_config
# gets its cutoff from the channel's power spectrum on the TRAINING rows only:
# the lowest frequency containing AUTO_CUTOFF_POWER_FRACTION of the signal's
# (mean-removed) power, clipped to [AUTO_CUTOFF_MIN_HZ, AUTO_CUTOFF_MAX_HZ].
# Rationale: a cutoff below the signal's own bandwidth smears real transients
# (exactly what anomaly detection needs); a cutoff above it removes nothing.
# The derived cutoffs are saved to models/filter_params.json when training and
# LOADED (never recomputed) at inference, exactly like the scaler.
AUTO_CUTOFF_POWER_FRACTION = 0.9999
AUTO_CUTOFF_MIN_HZ = 0.5      # floor: very slow channels should not be over-smoothed
AUTO_CUTOFF_MAX_HZ = 5.0      # ceiling: keeps denoising effective even if the
                              # fitting data itself is noisy (noise inflates bandwidth)
AUTO_CUTOFF_FALLBACK_HZ = 2.5 # used only if a channel is constant in the fitting rows
AUTO_CUTOFF_WELCH_NPERSEG = 4096
BUTTERWORTH_ORDER = 2
NYQUIST_GUARD = 0.99          # normalized cutoff never exceeds this fraction of Nyquist
ROLLING_MEDIAN_MIN_PERIODS = 1
FILTER_PARAMS_FILENAME = 'filter_params.json'   # saved under models/

# ============================================================================
# ROLLING FEATURE WINDOWS (Phase 2)
# ============================================================================
ROLLING_MEAN_MIN_PERIODS = 1
ROLLING_STD_MIN_PERIODS = 2

# ============================================================================
# INPUT-RATE CHECK (inference)
# ============================================================================
# Sequences/windows are counted in samples, so a file sampled at a different
# rate than the training data is rejected instead of silently mis-scaled.
SAMPLING_RATE_TOLERANCE_FRACTION = 0.02

# ============================================================================
# ATTRIBUTION (Phase 5)
# ============================================================================
ATTRIBUTION_TOP_N = 3
ATTRIBUTION_MIN_THRESHOLD_FLOOR = 1e-4

# ============================================================================
# NOISE ESTIMATE (Phase 1 report)
# ============================================================================
# Robust high-frequency noise estimate per channel: sigma = MAD(second
# difference) * NOISE_MAD_SCALE / sqrt(6). 1.4826 makes MAD consistent with
# a Gaussian standard deviation (a mathematical constant, not a tuning value).
NOISE_MAD_SCALE = 1.4826

# ============================================================================
# CORRECTION FLAGGING (Phase 1)
# ============================================================================
# A value outside its limits is always clipped, but it is only FLAGGED as
# "invalid/corrected" if it exceeds the limit by more than this fraction of
# the channel's (filtered) dynamic range in the file. Below that it is
# floating-point / solver noise (e.g. a simulated 0-100% signal that reaches
# 100.00000002), not a data-quality event. Unit-free by construction.
CORRECTION_TOLERANCE_FRACTION_OF_RANGE = 1e-6

# ============================================================================
# FILTER INITIAL CONDITIONS (Phase 1)
# ============================================================================
# True  : each causal Butterworth filter starts at steady state for the first
#         sample (scipy lfilter_zi * x[0]). Uses only that first sample, so it
#         is still causal and works identically in a live stream.
# False : start from zero state. This makes every channel with a large DC
#         offset (temperatures in K, pressures in Pa) ramp up from 0, a fake
#         start-up transient that takes seconds to tens of seconds to decay.
FILTER_STEADY_STATE_INIT = True

# ============================================================================
# TRAINING CALLBACKS (Phase 4)
# ============================================================================
EARLY_STOPPING_PATIENCE = 15
LR_REDUCTION_FACTOR = 0.5
LR_REDUCTION_PATIENCE = 5
LR_MIN = 1e-6

# ============================================================================
# INFERENCE
# ============================================================================
# Intermediate files written while scoring a NEW file go here, so they can
# never overwrite the training-side data/processed and data/features that the
# Phase 5 analysis scripts read together with the training metadata.
INFERENCE_WORK_DIR = 'data/inference'
INFERENCE_CHUNK_SEQUENCES = 2048   # windows materialised at once (memory bound)

# ============================================================================
# BOOTSTRAP
# ============================================================================
BOOTSTRAP_SEED = 42
