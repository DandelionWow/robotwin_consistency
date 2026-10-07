#!/usr/bin/env python3
"""Aggregate GT-history records with pair_id as the statistical unit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import PAIRWISE_METRICS, aggregate_paired_metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics", nargs="+", choices=PAIRWISE_METRICS, default=("psnr", "ssim"))
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = {
        "statistical_unit": "pair_id",
        "hierarchy": "repeat -> window-side -> pair-side -> expert-policy gap -> paired bootstrap",
        "metrics": {
            metric: aggregate_paired_metric(
                records,
                metric,
                bootstrap_samples=args.bootstrap_samples,
                confidence=args.confidence,
                seed=args.seed + index,
            )
            for index, metric in enumerate(args.metrics)
        },
        "jepa": "excluded: set-level auxiliary only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
