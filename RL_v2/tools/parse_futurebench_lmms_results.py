#!/usr/bin/env python3
"""Extract FutureBench metrics from lmms-eval aggregated results JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _find_results_json(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_futurebench_metrics(output_dir: str | Path, task: str = "futurebench_future_l1") -> dict[str, float]:
    root = Path(output_dir)
    results_path = _find_results_json(root)
    if results_path is None:
        return {}

    with results_path.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    task_results = payload.get("results", {}).get(task, {})
    if not task_results:
        # Fallback: first task block in results.
        results_block = payload.get("results", {})
        if isinstance(results_block, dict) and results_block:
            task_results = next(iter(results_block.values()))

    metrics: dict[str, float] = {}
    for key, value in task_results.items():
        if not isinstance(value, (int, float)):
            continue
        # lmms-eval keys look like "future_acc,none" -> future_acc
        metric_name = str(key).split(",", 1)[0]
        metrics[f"futurebench/{metric_name}"] = float(value)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="lmms-eval --output_path directory for one checkpoint")
    parser.add_argument("--task", default="futurebench_future_l1")
    parser.add_argument("--json", action="store_true", help="print metrics as JSON")
    args = parser.parse_args()

    metrics = parse_futurebench_metrics(args.output_dir, task=args.task)
    if not metrics:
        print(f"No FutureBench metrics found under {args.output_dir}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        for k, v in sorted(metrics.items()):
            print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
