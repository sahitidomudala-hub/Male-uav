"""
Onboard alert generation -- what happens after detection + attribution,
with no UI and no dependency on GCS or Secure Telemetry being reachable.

SCOPE, STATED EXPLICITLY
    This module produces DATA for downstream consumption. It does NOT:
    - decide or trigger any flight-control action (throttle, RTB, shutdown)
    - assume any particular downstream consumer exists or is reachable
    - claim any validated relationship to real engine faults
    Those are outside an anomaly-detection module's responsibility and
    require a certified flight-safety system, not this code.

WHAT THIS ACTUALLY DOES (the real "then what")
    1. DURABLE LOCAL LOG (onboard_alert_log.jsonl)
       Every detected anomaly is appended, one JSON object per line, to a
       local, append-only log file. This happens regardless of whether any
       telemetry link exists. JSON Lines format is used deliberately: it is
       crash-safe (a partially-written last line doesn't corrupt earlier
       ones) and can be tailed/streamed incrementally, unlike a single JSON
       array file which must be fully rewritten each time.

    2. COMPACT, PRIORITIZED ALERT RECORDS (priority_alerts.jsonl)
       Instead of treating all detected anomalies as equally urgent (which
       would waste a constrained telemetry link), each alert is tagged with
       an uplink priority based on severity:
           HIGH   -> immediate_uplink = True  (small enough to send even
                     over a degraded/low-bandwidth link)
           MEDIUM/LOW -> immediate_uplink = False (stays in the local log
                     for later bulk retrieval when link quality is good)
       This is the actual "don't wait for GCS" behavior: the aircraft
       decides locally, in real time, what's worth spending limited
       bandwidth on -- GCS/Secure Telemetry only need to handle the
       high-priority trickle immediately, not the full stream.

    3. COMPACT RECORD SCHEMA (deliberately small)
       Each alert record is a handful of scalar fields, not a full sequence
       sequence or CSV row -- this is what would actually be sent over a
       telemetry link, analogous to a DTC-style fault code rather than a
       full data dump:
           {
             "timestamp_start": <float, seconds>,
             "timestamp_end": <float, seconds>,
             "sequence_index": <int>,
             "anomaly_score": <float>,       # reconstruction_error / threshold
             "severity": "LOW"|"MEDIUM"|"HIGH",
             "top_channel": <str>,
             "top_channel_deviation": <float>,  # times over that channel's own threshold
             "immediate_uplink": <bool>
           }

WHAT THIS DOES NOT DO
    - Does not actually transmit anything over any network -- that is
      Secure Telemetry's (Layer 3's) job, not this module's. This script
      produces the data that layer would consume.
    - Does not deduplicate or debounce consecutive overlapping sequences
      that describe the same underlying event (e.g. the real t~621s event
      spans many overlapping sequences) -- see NOTE below.

NOTE ON OVERLAPPING SEQUENCES
    Because sequences use stride=1, a single ~2-second real event typically
    triggers many consecutive anomalous sequences (e.g. the known event
    spans dozens of sequence indices). Left unmerged, this would generate
    dozens of near-duplicate alerts for one real event. This script groups
    CONSECUTIVE anomalous sequence indices into a single alert covering
    their combined time span, so one real event produces one alert record,
    not one per overlapping window.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import pipeline_config as _pc
import schema_guard as _sg

# Severity tiers and the uplink policy live in pipeline_config.py.
SEVERITY_THRESHOLDS = _pc.SEVERITY_THRESHOLDS

# Only HIGH severity is flagged for immediate uplink under a constrained
# link. Adjust this policy here if your Secure Telemetry teammate wants a
# different cutoff -- it is a single, clearly-isolated decision point.
IMMEDIATE_UPLINK_SEVERITIES = _pc.IMMEDIATE_UPLINK_SEVERITIES


def severity_for_ratio(ratio: float) -> str:
    for upper, label in SEVERITY_THRESHOLDS:
        if ratio < upper:
            return label
    return SEVERITY_THRESHOLDS[-1][1]


def group_consecutive_sequences(anomaly_seq_indices: list) -> list:
    """Groups consecutive integers into runs, e.g. [301,302,303,310,311]
    -> [[301,302,303],[310,311]]. Used to merge overlapping-window
    detections of the same underlying event into one alert."""
    if not anomaly_seq_indices:
        return []
    sorted_idx = sorted(anomaly_seq_indices)
    groups = [[sorted_idx[0]]]
    for idx in sorted_idx[1:]:
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def build_alerts(results_df: pd.DataFrame, attribution_df: pd.DataFrame,
                  cleaned_df: pd.DataFrame, test_start_row: int, window: int, stride: int = 1) -> list:
    merged = results_df.merge(attribution_df, on="sequence_index", how="left")
    anomalies = merged[merged["classification"] == "Anomaly"].copy()

    groups = group_consecutive_sequences(anomalies["sequence_index"].tolist())

    alerts = []
    for group in groups:
        group_rows = anomalies[anomalies["sequence_index"].isin(group)]
        # Represent the group by its single worst (highest-ratio) sequence
        # for severity/attribution, but span the group's full time range.
        group_rows = group_rows.copy()
        group_rows["ratio"] = group_rows["reconstruction_error"] / group_rows["threshold"]
        worst = group_rows.loc[group_rows["ratio"].idxmax()]

        first_seq, last_seq = min(group), max(group)
        # Sequence i covers cleaned rows [test_start_row + i*stride, ... + window - 1]
        start_row = test_start_row + first_seq * stride
        end_row = test_start_row + last_seq * stride + window - 1   # last covered sample
        timestamp_start = float(cleaned_df.iloc[start_row]["Time"])
        timestamp_end = float(cleaned_df.iloc[min(end_row, len(cleaned_df) - 1)]["Time"])

        severity = severity_for_ratio(worst["ratio"])
        alert = {
            "timestamp_start": round(timestamp_start, 3),
            "timestamp_end": round(timestamp_end, 3),
            "sequence_index_range": [int(first_seq), int(last_seq)],
            "anomaly_score": round(float(worst["ratio"]), 3),
            "severity": severity,
            "top_channel": worst.get("top1_channel", None),
            "top_channel_deviation": round(float(worst.get("top1_times_over_threshold", 0)), 2)
                                       if pd.notna(worst.get("top1_times_over_threshold")) else None,
            "immediate_uplink": severity in IMMEDIATE_UPLINK_SEVERITIES,
        }
        alerts.append(alert)

    return alerts


def append_jsonl(records: list, path: Path):
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main(base_dir: str = "."):
    base = Path(base_dir)
    results_path = base / "outputs" / "anomaly_detection_results.csv"
    attribution_path = base / "outputs" / "anomaly_attribution.csv"
    cleaned_path = base / "data" / "processed" / "cleaned_telemetry.csv"
    metadata_path = base / "data" / "sequences" / "metadata.json"

    for p in [results_path, attribution_path, cleaned_path, metadata_path]:
        if not p.exists():
            sys.exit(f"Missing required file: {p}")

    results_df = pd.read_csv(results_path)
    attribution_df = pd.read_csv(attribution_path)
    cleaned_df = pd.read_csv(cleaned_path)
    meta = _sg.verify_schema(base)
    # first cleaned row of the test split = warm-up rows + train rows + val rows
    test_start_row = int(meta["warmup_rows_removed"]) + meta["train_rows"] + meta["val_rows"]

    alerts = build_alerts(results_df, attribution_df, cleaned_df, test_start_row,
                          window=int(meta["sequence_length"]), stride=int(meta["stride"]))

    log_path = base / "outputs" / "onboard_alert_log.jsonl"
    priority_path = base / "outputs" / "priority_alerts.jsonl"

    append_jsonl(alerts, log_path)
    immediate = [a for a in alerts if a["immediate_uplink"]]
    append_jsonl(immediate, priority_path)

    print(f"Grouped {len(results_df[results_df['classification']=='Anomaly'])} anomalous sequences "
          f"into {len(alerts)} distinct events (consecutive-run merged).")
    print(f"  Severity breakdown: "
          f"{sum(a['severity']=='HIGH' for a in alerts)} HIGH, "
          f"{sum(a['severity']=='MEDIUM' for a in alerts)} MEDIUM, "
          f"{sum(a['severity']=='LOW' for a in alerts)} LOW")
    print(f"  {len(immediate)} flagged for immediate uplink (HIGH severity)")
    print(f"  ✓ Full alert log: {log_path}")
    print(f"  ✓ Priority (immediate-uplink) queue: {priority_path}")

    return alerts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate onboard alert log and priority uplink queue")
    parser.add_argument("--base-dir", type=str, default=".")
    args = parser.parse_args()
    main(args.base_dir)
