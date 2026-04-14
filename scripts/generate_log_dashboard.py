#!/usr/bin/env python3
"""Read detections.log JSON lines and write a minimal static HTML summary."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/detection_dashboard.html"))
    args = ap.parse_args()
    preds = Counter()
    alerts = 0
    lines = 0
    if args.log.is_file():
        with open(args.log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lines += 1
                preds[o.get("pred", "?")] += 1
                if o.get("alert"):
                    alerts += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Detection summary</title></head>
<body><h1>Detection log summary</h1><p>Source: {args.log}</p>
<p>Lines: {lines}</p><p>Alerts: {alerts}</p><pre>{dict(preds)}</pre></body></html>""",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
