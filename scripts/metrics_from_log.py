#!/usr/bin/env python3
"""Summarize detection JSON lines (pred counts, mean p_ddos when alert)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> int:
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1
    preds = Counter()
    alerts = 0
    p_sum = 0.0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            preds[o.get("pred", "?")] += 1
            if o.get("alert"):
                alerts += 1
            if "p_ddos" in o:
                p_sum += float(o["p_ddos"])
                n += 1
    print(f"File: {path}")
    print(f"Lines parsed: {sum(preds.values())}")
    print("pred counts:", dict(preds))
    print(f"alert=True lines: {alerts}")
    if n:
        print(f"mean p_ddos: {p_sum / n:.6f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, nargs="?", default=Path("/var/log/synflood-detector/detections.log"))
    args = ap.parse_args()
    return summarize(args.log)


if __name__ == "__main__":
    raise SystemExit(main())
