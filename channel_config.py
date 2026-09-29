"""
channel_config.py
==================
SINGLE SOURCE OF TRUTH for per-channel telemetry handling.

Every phase (1: preprocessing/correction/filtering, 2: feature extraction,
downstream: handoff CSV) reads its per-channel behavior from this one table
instead of maintaining separate hardcoded lists. To add, remove, or change a
telemetry channel, edit ONLY this file.

To add a new channel:
    1. Add a row to CHANNEL_CONFIG below.
    2. Decide 'correction': does this channel have physically-impossible
       values that need clipping/repair? Use None or a tuple like
       ('clip_min', {'min': 0}) or ('clip_range', {'min': 0, 'max': 100}).
    3. Decide 'moving_stats' / 'rate': does a 1-second moving average/std or
       a rate-of-change feature make physical sense for this signal?
    4. Decide 'filter': filter type and parameters, or 'none'.
    5. Decide 'model_input': should this channel's features go to the LSTM?
       (Altitude and BaroPress are passthrough-only for context.)
    6. Provide 'unit' string for documentation and handoff files.
    7. Retrain the model if you changed which channels have model_input=True
       (changing the feature set changes the LSTM's input dimension).

FILTER RATIONALE:
    - Butterworth low-pass: for signals whose real physical bandwidth is
      genuinely limited (thermal channels, pressure dynamics).
    - Causal median filter: for signals prone to single-sample spike noise
      (potentiometer bounce, pulse miscounts).
    - All filters are CAUSAL (use only past/current samples) for real-time
      inference and to maintain valid sequence-to-time mapping.
    - Cutoffs are in Hz and adapt automatically to detected sampling rate.
    - ('butterworth', 'auto'): the cutoff is DERIVED from the channel's own
      bandwidth on the training rows (see pipeline_config AUTO_CUTOFF_*), saved
      to models/filter_params.json and reused, never recomputed, at inference.
      A fixed number of Hz is also accepted: ('butterworth', 2.5).
    - The current DAQ data is a noise-free Simulink simulation, so the amount
      of noise these filters remove on real noisy data is NOT validated by
      this data; outputs/noise_sweep.csv measures it.
"""

# Correction rules registry
# 'clip_min': clip values below min to min
# 'clip_max': clip values above max to max
# 'clip_range': clip to [min, max]

CHANNEL_CONFIG = {
    # channel      correction                   moving_stats  rate   filter              model_input  unit
    'EngTrq':    {'correction': None,                 'moving_stats': True,  'rate': True,  'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'N*m'},
    'RPM':       {'correction': None,                 'moving_stats': True,  'rate': True,  'filter': ('median', 5),         'model_input': True,  'unit': 'rpm'},
    'MAT':       {'correction': None,                 'moving_stats': False, 'rate': False, 'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'K'},
    'MAP':       {'correction': None,                 'moving_stats': True,  'rate': False, 'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'Pa abs'},
    'EGT':       {'correction': None,                 'moving_stats': True,  'rate': True,  'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'K'},
    'FuelFlow':  {'correction': ('clip_min', {'min': 0}), 'moving_stats': True,  'rate': False, 'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'kg/s (inferred)'},
    'CH':        {'correction': None,                 'moving_stats': False, 'rate': False, 'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'K'},
    'BaroPress': {'correction': None,                 'moving_stats': False, 'rate': False, 'filter': ('median', 5),         'model_input': False, 'unit': 'Pa abs'},
    'TurboSpd':  {'correction': None,                 'moving_stats': False, 'rate': False, 'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'rad/s'},
    'TPS':       {'correction': ('clip_range', {'min': 0, 'max': 100}), 'moving_stats': False, 'rate': True,  'filter': ('median', 5),         'model_input': True,  'unit': '% (inferred)'},
    'OilPress':  {'correction': None,                 'moving_stats': True,  'rate': False, 'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'Pa abs'},
    'OilTemp':   {'correction': None,                 'moving_stats': False, 'rate': False, 'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'K'},
    'AltCurrent':{'correction': None,                 'moving_stats': False, 'rate': False, 'filter': ('median', 5),         'model_input': True,  'unit': 'A'},
    'BusVoltage':{'correction': None,                 'moving_stats': True,  'rate': False, 'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'V'},
    'InjTiming': {'correction': None,                 'moving_stats': False, 'rate': False, 'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'deg crank angle'},
    'Altitude':  {'correction': None,                 'moving_stats': False, 'rate': False, 'filter': ('none', None),         'model_input': False, 'unit': 'm'},
    'Vib0p5X':   {'correction': None,                 'moving_stats': False, 'rate': False, 'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'g (estimate)'},
    'Vib1X':     {'correction': None,                 'moving_stats': True,  'rate': False, 'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'g (estimate)'},
    'Vib2X':     {'correction': None,                 'moving_stats': False, 'rate': False, 'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'g (estimate)'},
    'VibRMS':    {'correction': None,                 'moving_stats': True,  'rate': False, 'filter': ('butterworth', 'auto'),  'model_input': True,  'unit': 'g (estimate)'},
}

# Fixed, non-channel column. Never filtered, never a feature source.
TIME_COLUMN = 'Time'

REQUIRED_COLUMNS = [TIME_COLUMN] + list(CHANNEL_CONFIG.keys())


def get_channels_with_correction():
    """Returns {channel_name: (rule, params)} for channels needing repair."""
    return {ch: cfg['correction'] for ch, cfg in CHANNEL_CONFIG.items() if cfg['correction']}


def get_moving_stat_channels():
    """Channels that get a 1s trailing moving average + moving std feature."""
    return [ch for ch, cfg in CHANNEL_CONFIG.items() 
            if cfg.get('moving_stats', False) and cfg.get('model_input', True)]


def get_rate_channels():
    """Channels that get a causal rate-of-change (diff) feature."""
    return [ch for ch, cfg in CHANNEL_CONFIG.items() 
            if cfg.get('rate', False) and cfg.get('model_input', True)]


def get_raw_only_channels():
    """Channels that get only a raw feature (no moving stats, no rate), among model_input channels."""
    return [ch for ch, cfg in CHANNEL_CONFIG.items()
            if cfg.get('model_input', True) and not cfg.get('moving_stats', False) and not cfg.get('rate', False)]


def get_filter_config(channel_config=None):
    """Returns {channel_name: (filter_type, param) or 'none'} for Phase 1 filtering.
    filter_type is 'butterworth', 'median' or 'none'."""
    cfg_all = CHANNEL_CONFIG if channel_config is None else channel_config
    return {ch: cfg['filter'] if cfg['filter'] != 'none' else ('none', None) 
            for ch, cfg in cfg_all.items()}


def get_physical_channels():
    """All physical telemetry channel names, in config order."""
    return list(CHANNEL_CONFIG.keys())


def get_model_input_channels():
    """Channels whose features are fed to the model (model_input=True)."""
    return [ch for ch, cfg in CHANNEL_CONFIG.items() if cfg.get('model_input', True)]


def get_passthrough_channels():
    """Channels excluded from model but kept in cleaned/handoff (model_input=False)."""
    return [ch for ch, cfg in CHANNEL_CONFIG.items() if not cfg.get('model_input', True)]


def get_units_dict():
    """Returns {channel_name: unit_string} for all channels."""
    return {ch: cfg['unit'] for ch, cfg in CHANNEL_CONFIG.items()}
