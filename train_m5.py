"""Validate M5 inputs and generate isolated Stage 2 feature shards."""
from __future__ import annotations

import json
import sys
import argparse

from m5_pipeline.validation import validate_m5_data
from m5_pipeline.features import generate_feature_dataset
from m5_pipeline.training import train_and_evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validate-only", action="store_true", help="validate M5 inputs only")
    group.add_argument("--features-only", action="store_true", help="validate inputs and generate Stage 2 features")
    group.add_argument("--train", action="store_true", help="train/evaluate Stage 3 models from existing feature shards")
    args = parser.parse_args()
    try:
        if args.train:
            result = train_and_evaluate()
            print(json.dumps(result, indent=2))
            return 0
        validation = validate_m5_data()
        if args.validate_only:
            print(json.dumps(validation, indent=2))
            return 0
        features = generate_feature_dataset()
    except Exception as exc:
        print(f"M5 pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"validation": validation, "features": features}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
