import argparse
import math
import struct
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def iter_pcap_packets_with_timestamp(path):
    """
    Yield (timestamp_unix_float, raw_packet_bytes) for each packet in classic pcap.
    Timestamp is seconds since Unix epoch (UTC), including fractional microseconds.
    """
    with open(path, "rb") as f:
        global_header = f.read(24)
        if len(global_header) != 24:
            return

        magic_number = struct.unpack("I", global_header[:4])[0]
        if magic_number == 0xA1B2C3D4:
            endian = "<"
        elif magic_number == 0xD4C3B2A1:
            endian = ">"
        else:
            raise ValueError("Unsupported pcap magic number")

        hdr_struct = struct.Struct(f"{endian}IIII")
        while True:
            header = f.read(16)
            if len(header) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = hdr_struct.unpack(header)
            data = f.read(incl_len)
            if len(data) < incl_len:
                break
            ts = float(ts_sec) + float(ts_usec) / 1_000_000.0
            yield ts, data


def iter_pcap_packets(path):
    """Yield raw packet bytes from a pcap file (little/big endian)."""
    for _ts, data in iter_pcap_packets_with_timestamp(path):
        yield data


def is_tcp_syn(packet_bytes):
    """Lightweight TCP SYN check without scapy."""
    # Minimum Ethernet + IPv4 + TCP header length: 14 + 20 + 20
    if len(packet_bytes) < 54:
        return False

    offset = 12
    eth_type = int.from_bytes(packet_bytes[offset:offset + 2], "big")

    # VLAN tagged frames (802.1Q)
    if eth_type == 0x8100 and len(packet_bytes) >= 18:
        eth_type = int.from_bytes(packet_bytes[16:18], "big")
        ip_start = 18
    else:
        ip_start = 14

    if eth_type != 0x0800:  # IPv4
        return False

    if len(packet_bytes) < ip_start + 20:
        return False

    ihl = (packet_bytes[ip_start] & 0x0F) * 4
    if ihl < 20:
        return False

    proto = packet_bytes[ip_start + 9]
    if proto != 6:  # TCP
        return False

    tcp_start = ip_start + ihl
    if len(packet_bytes) < tcp_start + 14:  # need up to flags byte
        return False

    flags = packet_bytes[tcp_start + 13]
    return (flags & 0x02) != 0  # SYN bit


def count_eligible_packets(path: Path, syn_only: bool) -> int:
    """How many packets would be considered after syn_only filter."""
    n = 0
    for pkt in iter_pcap_packets(path):
        if syn_only and not is_tcp_syn(pkt):
            continue
        n += 1
    return n


def count_eligible_groups(
    n_packets: int, packets_per_image: int, include_partial_tail: bool
) -> int:
    """Number of composite images (non-overlapping groups along the eligible stream)."""
    if n_packets <= 0 or packets_per_image <= 0:
        return 0
    if packets_per_image == 1:
        return n_packets
    full = n_packets // packets_per_image
    if include_partial_tail and (n_packets % packets_per_image) != 0:
        return full + 1
    return full


def stratified_indices(n: int, k: int) -> list[int]:
    """
    Pick k indices in [0, n-1] spread across the stream (centers of k strata).
    Used so large PCAPS (~200MB) contribute images from early/mid/late capture, not only the head.
    When packets_per_image>1, n is the number of *groups*, not packets.
    """
    k = min(k, n)
    if k <= 0:
        return []
    if k == 1:
        return [n // 2]
    return [min(n - 1, int((j + 0.5) * n / k)) for j in range(k)]


def raw_bytes_to_image_224x224(raw: bytes):
    """
    Map arbitrary raw bytes (one or many packets concatenated) to 224x224 grayscale
    using the same layout as the paper (square pad/crop then resize).
    """
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)

    # Form 2D grayscale image: smallest square that fits, pad to square
    side = math.ceil(math.sqrt(len(arr)))
    target = side * side
    if len(arr) < target:
        arr = np.pad(arr, (0, target - len(arr)), mode="constant", constant_values=0)
    else:
        arr = arr[:target]

    img_2d = arr.reshape((side, side))

    pil_img = Image.fromarray(img_2d, mode="L")
    pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)

    return np.array(pil_img)


def packet_to_image_224x224(pkt_bytes, syn_only):
    """
    Per paper (Section III.B): convert one packet to 224x224 grayscale.
    Used by offline training defaults and by run_pipeline.py for live windows.
    """
    if syn_only and not is_tcp_syn(pkt_bytes):
        return None
    return raw_bytes_to_image_224x224(pkt_bytes)


def _iter_eligible_packets(path: Path, syn_only: bool):
    for pkt in iter_pcap_packets(path):
        if syn_only and not is_tcp_syn(pkt):
            continue
        yield pkt


def _ensure_repo_on_path() -> None:
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def run_time_window_mode(
    pcap_path: Path,
    split_out_root: Path,
    prefix: str,
    window_sec: float,
    max_images: int,
    sample_strategy: str,
    cic_schedule: Path,
    syn_threshold: int,
    window_logic: str,
) -> int:
    """
    One image per time bucket (default 1s UTC): all packets in that bucket concatenated.
    Label = CIC schedule overlap with bucket AND (default) SYN count > threshold.
    Writes under split_out_root/ddos and split_out_root/normal.
    """
    import json

    _ensure_repo_on_path()
    from scripts.classify_pcap_syn import label_window_with_schedule_and_syn

    if window_sec <= 0:
        raise ValueError("window_sec must be positive")

    with open(cic_schedule, encoding="utf-8") as f:
        cfg = json.load(f)

    buckets: dict[int, list[bytes]] = defaultdict(list)
    syn_per_bucket: dict[int, int] = defaultdict(int)

    for ts, pkt in iter_pcap_packets_with_timestamp(pcap_path):
        bid = int(math.floor(ts / window_sec))
        buckets[bid].append(pkt)
        if is_tcp_syn(pkt):
            syn_per_bucket[bid] += 1

    bucket_ids = sorted(buckets.keys())
    if not bucket_ids:
        print(f"Saved 0 time-window images under {split_out_root}")
        return 1

    n_b = len(bucket_ids)
    want = min(max_images, n_b)
    if sample_strategy == "even":
        pos = stratified_indices(n_b, want)
        chosen_bucket_ids = [bucket_ids[i] for i in pos]
    else:
        chosen_bucket_ids = bucket_ids[:want]

    ddos_dir = split_out_root / "ddos"
    normal_dir = split_out_root / "normal"
    ddos_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for bid in chosen_bucket_ids:
        pkts = buckets[bid]
        raw = b"".join(pkts)
        img_arr = raw_bytes_to_image_224x224(raw)
        if img_arr is None:
            continue

        t0_unix = bid * window_sec
        t1_unix = (bid + 1) * window_sec
        t0_utc = datetime.fromtimestamp(t0_unix, tz=timezone.utc)
        t1_utc = datetime.fromtimestamp(t1_unix, tz=timezone.utc)
        sc = syn_per_bucket[bid]
        cls = label_window_with_schedule_and_syn(
            cfg,
            t0_utc,
            t1_utc,
            str(pcap_path),
            sc,
            syn_threshold,
            logic=window_logic,
        )
        out_dir = ddos_dir if cls == "ddos" else normal_dir
        fname = f"{prefix}_t{int(t0_unix)}_syn{sc}.png"
        Image.fromarray(img_arr, mode="L").save(out_dir / fname)
        saved += 1

    print(
        f"Saved {saved} time-window images ({window_sec}s buckets, logic={window_logic}) "
        f"under {split_out_root}/{{ddos,normal}}"
    )
    return 0 if saved > 0 else 1


def _flush_group(
    buf: list[bytes],
    out_dir: Path,
    prefix: str,
    saved: int,
) -> tuple[int, list[bytes]]:
    """Save one composite image from buf; return (new_saved, cleared buf)."""
    if not buf:
        return saved, buf
    raw = b"".join(buf)
    img_arr = raw_bytes_to_image_224x224(raw)
    if img_arr is None:
        return saved, []
    img = Image.fromarray(img_arr, mode="L")
    img.save(out_dir / f"{prefix}_{saved}.png")
    return saved + 1, []


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert PCAP(s) to 224x224 grayscale images. "
            "Optional: merge N consecutive eligible packets into one image to cover long captures; "
            "use even sampling over groups for large files."
        )
    )
    parser.add_argument("--pcap", required=True, help="Path to PCAP file.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (required for packet-group mode; not used for time-window mode).",
    )
    parser.add_argument("--prefix", help="Filename prefix. Defaults to PCAP basename.")
    parser.add_argument("--max-images", type=int, default=200, help="Maximum composite images to save.")
    parser.add_argument(
        "--packets-per-image",
        type=int,
        default=1,
        help="Concatenate this many consecutive eligible packets into one image (1 = one packet per image).",
    )
    parser.add_argument(
        "--no-partial-tail",
        action="store_true",
        help="Drop trailing packets that do not fill a full group (only applies when packets-per-image > 1).",
    )
    parser.add_argument(
        "--syn-only",
        action="store_true",
        help="Only consider TCP SYN packets (skip others for counting and grouping).",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["sequential", "even"],
        default="even",
        help="sequential: first max-images groups in order. even: stratified sample over group index.",
    )
    parser.add_argument(
        "--time-window-seconds",
        type=float,
        default=0.0,
        help="If >0: one image per time bucket (UTC-based); all packets in [t,t+W) concatenated. "
        "Uses --split-out-root and --cic-schedule; labels each bucket with schedule + SYN count.",
    )
    parser.add_argument(
        "--split-out-root",
        type=Path,
        default=None,
        help="Directory that will contain ddos/ and normal/ (required when --time-window-seconds > 0).",
    )
    parser.add_argument(
        "--cic-schedule",
        type=Path,
        default=None,
        help="Path to cic_ddos2019_syn_windows.json (required when --time-window-seconds > 0).",
    )
    parser.add_argument(
        "--window-syn-threshold",
        type=int,
        default=100,
        help="SYN count in bucket must exceed this for ddos when using default --window-label-logic and.",
    )
    parser.add_argument(
        "--window-label-logic",
        choices=["and", "or"],
        default="and",
        help="and: ddos if schedule overlaps bucket AND syn_count>threshold. or: either condition.",
    )
    args = parser.parse_args()

    pcap_path = Path(args.pcap)
    if not pcap_path.is_file():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    if args.time_window_seconds and args.time_window_seconds > 0:
        if args.split_out_root is None or args.cic_schedule is None:
            raise SystemExit(
                "Time-window mode requires --split-out-root and --cic-schedule "
                "(parent of train/ddos & train/normal, e.g. .../images/train)."
            )
        if not args.cic_schedule.is_file():
            raise SystemExit(f"CIC schedule file not found: {args.cic_schedule}")
        prefix = args.prefix or pcap_path.stem
        return run_time_window_mode(
            pcap_path,
            args.split_out_root,
            prefix,
            float(args.time_window_seconds),
            args.max_images,
            args.sample_strategy,
            args.cic_schedule,
            int(args.window_syn_threshold),
            str(args.window_label_logic),
        )

    if args.out is None:
        raise SystemExit("Packet-group mode requires --out (or use --time-window-seconds > 0).")

    ppi = max(1, int(args.packets_per_image))
    include_partial = not args.no_partial_tail

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or pcap_path.stem

    saved = 0
    if args.sample_strategy == "sequential":
        buf: list[bytes] = []
        for pkt_bytes in _iter_eligible_packets(pcap_path, args.syn_only):
            buf.append(pkt_bytes)
            if len(buf) >= ppi:
                saved, buf = _flush_group(buf, out_dir, prefix, saved)
                if saved >= args.max_images:
                    break
        if buf and include_partial and saved < args.max_images:
            saved, _ = _flush_group(buf, out_dir, prefix, saved)
    else:
        n_pkt = count_eligible_packets(pcap_path, args.syn_only)
        n_groups = count_eligible_groups(n_pkt, ppi, include_partial)
        if n_groups == 0:
            print(f"Saved {saved} images (224x224) to {out_dir}")
            return 1
        want = min(args.max_images, n_groups)
        targets = frozenset(stratified_indices(n_groups, want))

        g_idx = 0
        buf = []
        for pkt_bytes in _iter_eligible_packets(pcap_path, args.syn_only):
            buf.append(pkt_bytes)
            if len(buf) >= ppi:
                if g_idx in targets:
                    img_arr = raw_bytes_to_image_224x224(b"".join(buf))
                    if img_arr is not None:
                        img = Image.fromarray(img_arr, mode="L")
                        img.save(out_dir / f"{prefix}_{saved}.png")
                        saved += 1
                buf = []
                g_idx += 1
                if saved >= want:
                    break
        if buf and include_partial and g_idx in targets and saved < want:
            img_arr = raw_bytes_to_image_224x224(b"".join(buf))
            if img_arr is not None:
                img = Image.fromarray(img_arr, mode="L")
                img.save(out_dir / f"{prefix}_{saved}.png")
                saved += 1

    print(f"Saved {saved} images (224x224) to {out_dir}")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
