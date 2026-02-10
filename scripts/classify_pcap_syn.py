#!/usr/bin/env python3
"""
Classify a PCAP as ddos or normal by SYN count.
Exits early once SYN count exceeds threshold (no need to read the rest of the file).
"""
import argparse
import sys
from pathlib import Path

# Allow importing from sibling script when run from any directory
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pcap_to_images import iter_pcap_packets, is_tcp_syn


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


def main():
    parser = argparse.ArgumentParser(
        description="Classify PCAP by SYN count (exit early if above threshold)."
    )
    parser.add_argument("pcap", help="Path to PCAP file.")
    parser.add_argument(
        "threshold",
        type=int,
        default=100,
        nargs="?",
        help="SYN count above which to classify as ddos (default 100).",
    )
    args = parser.parse_args()

    pcap_path = Path(args.pcap)
    if not pcap_path.is_file():
        print(f"Error: PCAP not found: {pcap_path}", file=sys.stderr)
        sys.exit(2)

    result = classify_pcap_by_syn(str(pcap_path), args.threshold)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
