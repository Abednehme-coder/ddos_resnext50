#!/usr/bin/env python3
"""Minimal one-line live demo view for detections.log (stdin JSONL). Use: ... | python3 -u ..."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def pct_ddos(p: object) -> str:
    if not isinstance(p, (int, float)):
        return "?"
    x = 100.0 * float(p)
    if x >= 10:
        return f"{x:.0f}%"
    return f"{x:.1f}%"


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = o.get("ts")
        if isinstance(ts, (int, float)):
            clock = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")
        else:
            clock = "??:??:??"
        raw = str(o.get("pred", "?")).lower()
        verdict = "DDOS" if raw == "ddos" else "NORMAL"
        ptxt = pct_ddos(o.get("p_ddos"))
        alert = o.get("alert")
        if alert is None:
            alert_txt = "—"
        else:
            alert_txt = "YES" if alert else "no"
        # One short line: easy to read on a projector
        print(f"{clock} UTC  |  {verdict:6}  |  DDoS score {ptxt}  |  alert {alert_txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
