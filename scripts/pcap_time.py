"""PCAP timestamp utilities (classic pcap). Used for CIC-DDoS2019 schedule-based labeling."""
from __future__ import annotations

import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


def read_pcap_time_span_utc(path: str | Path) -> Optional[Tuple[datetime, datetime]]:
    """
    Return (first_packet_utc, last_packet_utc) for a classic pcap file.
    Uses UTC from pcap ts_sec/ts_usec. Returns None if file is empty/invalid.
    """
    path = Path(path)
    if not path.is_file():
        return None

    with open(path, "rb") as f:
        global_header = f.read(24)
        if len(global_header) < 24:
            return None

        magic_number = struct.unpack("I", global_header[:4])[0]
        if magic_number == 0xA1B2C3D4:
            endian = "<"
        elif magic_number == 0xD4C3B2A1:
            endian = ">"
        else:
            return None

        hdr_struct = struct.Struct(f"{endian}IIII")
        first_ts: Optional[Tuple[int, int]] = None
        last_ts: Optional[Tuple[int, int]] = None

        while True:
            header = f.read(16)
            if len(header) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = hdr_struct.unpack(header)
            data = f.read(incl_len)
            if len(data) < incl_len:
                break
            if first_ts is None:
                first_ts = (ts_sec, ts_usec)
            last_ts = (ts_sec, ts_usec)

    if first_ts is None or last_ts is None:
        return None

    def to_utc(sec: int, usec: int) -> datetime:
        # Normalise usec to [0, 999999]
        frac = usec / 1_000_000.0
        return datetime.fromtimestamp(sec + frac, tz=timezone.utc)

    return (to_utc(*first_ts), to_utc(*last_ts))
