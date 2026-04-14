#!/usr/bin/env python3
"""Pretty-print detections.log JSON lines from stdin (use with: tail -f ... | python3 -u ...)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            print(f"[non-JSON] {line[:120]}", flush=True)
            continue
        ts = o.get("ts")
        if isinstance(ts, (int, float)):
            t = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            t = "?"
        pred = o.get("pred", "?")
        p = o.get("p_ddos")
        streak = o.get("streak", "-")
        alert = o.get("alert", False)
        nimg = o.get("images_all", "-")
        pcap = o.get("pcap", "")
        base = "-"
        if isinstance(pcap, str) and pcap:
            base = pcap.rsplit("/", 1)[-1]
            if len(base) > 36:
                base = base[:33] + "..."
        p_s = f"{float(p):.6f}" if isinstance(p, (int, float)) else str(p)
        print(
            f"{t}  {str(pred):6}  p_ddos={p_s}  streak={streak}  alert={alert}  imgs={nimg}  pcap={base}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
