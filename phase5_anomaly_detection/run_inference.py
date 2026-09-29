"""
Phase 5: standalone inference entry point.

Thin wrapper around inference_core.run_inference() -- the single shared
implementation also used by `run_pipeline.py --mode inference`. Uses the
trained artifacts only: NO retraining, refitting or threshold recalculation.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import inference_core


def run_inference(input_csv: str, base_dir: str = None, output_prefix: str = 'inference'):
    base = Path(base_dir).resolve() if base_dir else inference_core.ROOT
    return inference_core.run_inference(input_csv, base_dir=base, output_prefix=output_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with the trained artifacts')
    parser.add_argument('--input', required=True, help='Input telemetry CSV')
    parser.add_argument('--base-dir', default=None, help='Pipeline root (default: this repo)')
    parser.add_argument('--output-prefix', default='inference', help='Prefix for output files')
    args = parser.parse_args()
    sys.exit(0 if run_inference(args.input, args.base_dir, args.output_prefix) else 1)
