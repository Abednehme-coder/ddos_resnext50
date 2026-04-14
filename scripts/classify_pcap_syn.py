#!/usr/bin/env python3
"""
Classify a PCAP as ddos or normal.

- syn_count (default): TCP SYN count vs threshold (early exit).
- cic_schedule: overlap of PCAP time span with CIC-DDoS2019 SYN windows (see cic_ddos2019_syn_windows.json).
  Falls back to syn_count if timestamps/config cannot be applied.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Imports when run as script from repo
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pcap_time import read_pcap_time_span_utc  # noqa: E402

from scripts.pcap_to_images import iter_pcap_packets, is_tcp_syn  # noqa: E402


def classify_pcap_by_syn(pcap_path: str, threshold: int = 100) -> str:
    """
    Classify PCAP as 'ddos' or 'normal' based on TCP SYN count.
    Stops reading as soon as count > threshold (early exit for DDoS files).
    """
    count = 0
    for pkt in iter_pcap_packets(pcap_path):
        if is_tcp_syn(pkt):
            count += 1
            if count > threshold:
                return "ddos"
    return "normal"


def _parse_hhmm(s: str) -> tuple[int, int]:
    parts = s.strip().split(":")
    return int(parts[0]), int(parts[1])


def _local_window_on_date(
    day, start_str: str, end_str: str, tz
) -> tuple[datetime, datetime]:
    """Build [start, end] in tz on calendar date `day` (datetime.date)."""
    sh, sm = _parse_hhmm(start_str)
    eh, em = _parse_hhmm(end_str)
    start = datetime(
        day.year, day.month, day.day, sh, sm, 0, tzinfo=tz
    )
    end = datetime(day.year, day.month, day.day, eh, em, 0, tzinfo=tz)
    if end < start:
        end = end + timedelta(days=1)
    return start, end


def _spans_overlap(
    a0: datetime, a1: datetime, b0: datetime, b1: datetime
) -> bool:
    return a0 <= b1 and a1 >= b0


def _filename_to_dd_mm_yyyy(path: str) -> Optional[str]:
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", Path(path).name)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def _parse_dd_mm_yyyy(key: str):
    return datetime.strptime(key, "%d-%m-%Y").date()


def classify_pcap_cic_schedule(
    pcap_path: str,
    schedule_config_path: Path,
    syn_fallback_threshold: int,
) -> str:
    try:
        from zoneinfo import ZoneInfo

        with open(schedule_config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        tz = ZoneInfo(cfg["timezone"])
        date_to_day = cfg.get("dd_mm_yyyy_to_cic_day", {})
        syn_windows = cfg.get("syn_windows", {})

        span = read_pcap_time_span_utc(pcap_path)
        if span is None:
            return classify_pcap_by_syn(pcap_path, syn_fallback_threshold)

        t0_utc, t1_utc = span
        t0_local = t0_utc.astimezone(tz)
        t1_local = t1_utc.astimezone(tz)

        dd_key = _filename_to_dd_mm_yyyy(pcap_path)
        cic_day = None
        anchor_date = None

        if dd_key and dd_key in date_to_day:
            cic_day = date_to_day[dd_key]
            anchor_date = _parse_dd_mm_yyyy(dd_key)
        else:
            k = t0_local.strftime("%d-%m-%Y")
            if k in date_to_day:
                cic_day = date_to_day[k]
                anchor_date = t0_local.date()

        if cic_day is None or anchor_date is None:
            return classify_pcap_by_syn(pcap_path, syn_fallback_threshold)

        windows = syn_windows.get(cic_day, [])
        if not windows:
            return classify_pcap_by_syn(pcap_path, syn_fallback_threshold)

        for w in windows:
            w0, w1 = _local_window_on_date(
                anchor_date, w["start"], w["end"], tz
            )
            if _spans_overlap(t0_local, t1_local, w0, w1):
                return "ddos"
        return "normal"
    except Exception:
        return classify_pcap_by_syn(pcap_path, syn_fallback_threshold)


def schedule_overlaps_utc_window(
    cfg: dict,
    t0_utc: datetime,
    t1_utc: datetime,
    pcap_path: str,
) -> bool:
    """
    True if [t0_utc, t1_utc] (inclusive-ish; use same overlap rule as PCAP span) intersects
    any CIC SYN window in cfg for the anchor day of this PCAP / window.
    """
    try:
        from zoneinfo import ZoneInfo

        if t0_utc.tzinfo is None:
            t0_utc = t0_utc.replace(tzinfo=timezone.utc)
        if t1_utc.tzinfo is None:
            t1_utc = t1_utc.replace(tzinfo=timezone.utc)

        tz = ZoneInfo(cfg["timezone"])
        date_to_day = cfg.get("dd_mm_yyyy_to_cic_day", {})
        syn_windows = cfg.get("syn_windows", {})

        t0_local = t0_utc.astimezone(tz)
        t1_local = t1_utc.astimezone(tz)

        dd_key = _filename_to_dd_mm_yyyy(pcap_path)
        cic_day = None
        anchor_date = None

        if dd_key and dd_key in date_to_day:
            cic_day = date_to_day[dd_key]
            anchor_date = _parse_dd_mm_yyyy(dd_key)
        else:
            k = t0_local.strftime("%d-%m-%Y")
            if k in date_to_day:
                cic_day = date_to_day[k]
                anchor_date = t0_local.date()

        if cic_day is None or anchor_date is None:
            return False

        windows = syn_windows.get(cic_day, [])
        for w in windows:
            w0, w1 = _local_window_on_date(anchor_date, w["start"], w["end"], tz)
            if _spans_overlap(t0_local, t1_local, w0, w1):
                return True
        return False
    except Exception:
        return False


def label_window_with_schedule_and_syn(
    cfg: dict,
    t0_utc: datetime,
    t1_utc: datetime,
    pcap_path: str,
    syn_count: int,
    syn_threshold: int,
    logic: str = "and",
) -> str:
    """
    Per-second (or arbitrary) window label for training/detection.

    Default logic 'and': ddos only if the window overlaps the CIC SYN schedule *and*
    SYN count in that window exceeds syn_threshold.

    logic 'or': ddos if either schedule overlap or SYN count (use with care).
    """
    in_sched = schedule_overlaps_utc_window(cfg, t0_utc, t1_utc, pcap_path)
    syn_high = syn_count > syn_threshold
    if logic == "or":
        return "ddos" if (in_sched or syn_high) else "normal"
    return "ddos" if (in_sched and syn_high) else "normal"


def main() -> int:
    default_method = os.environ.get("CLASSIFY_METHOD", "syn_count")
    default_schedule = os.environ.get("CIC_SCHEDULE_CONFIG")
    if default_schedule:
        default_schedule_path = Path(default_schedule)
    else:
        default_schedule_path = _SCRIPT_DIR / "cic_ddos2019_syn_windows.json"

    parser = argparse.ArgumentParser(
        description="Classify PCAP by SYN count or CIC-DDoS2019 SYN schedule."
    )
    parser.add_argument("pcap", help="Path to PCAP file.")
    parser.add_argument(
        "threshold",
        type=int,
        nargs="?",
        default=100,
        help="SYN count above which to classify as ddos when using syn_count or fallback (default 100).",
    )
    parser.add_argument(
        "--method",
        choices=["syn_count", "cic_schedule"],
        default=default_method,
        help="Labeling method (default: env CLASSIFY_METHOD or syn_count).",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=default_schedule_path,
        help="JSON config with SYN windows (default: env CIC_SCHEDULE_CONFIG or scripts/cic_ddos2019_syn_windows.json).",
    )
    args = parser.parse_args()

    pcap_path = Path(args.pcap)
    if not pcap_path.is_file():
        print(f"Error: PCAP not found: {pcap_path}", file=sys.stderr)
        return 2

    if args.method == "cic_schedule":
        if not args.schedule.is_file():
            print(
                f"Warning: schedule file not found: {args.schedule}, using syn_count",
                file=sys.stderr,
            )
            result = classify_pcap_by_syn(str(pcap_path), args.threshold)
        else:
            result = classify_pcap_cic_schedule(
                str(pcap_path), args.schedule, args.threshold
            )
    else:
        result = classify_pcap_by_syn(str(pcap_path), args.threshold)

    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
