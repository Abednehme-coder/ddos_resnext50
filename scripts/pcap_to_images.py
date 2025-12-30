import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def iter_pcap_packets(path):
    """Yield raw packet bytes from a pcap file (little/big endian)."""
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


def packet_to_image(pkt_bytes, img_size, syn_only):
    if syn_only and not is_tcp_syn(pkt_bytes):
        return None

    arr = np.frombuffer(pkt_bytes, dtype=np.uint8)
    target_size = img_size * img_size

    if len(arr) < target_size:
        arr = np.pad(arr, (0, target_size - len(arr)))
    else:
        arr = arr[:target_size]

    return arr.reshape((img_size, img_size))


def main():
    parser = argparse.ArgumentParser(description="Convert PCAP packets to grayscale images.")
    parser.add_argument("--pcap", required=True, help="Path to PCAP file.")
    parser.add_argument("--out", required=True, help="Output directory for images.")
    parser.add_argument("--prefix", help="Filename prefix. Defaults to PCAP basename.")
    parser.add_argument("--max-images", type=int, default=200, help="Maximum images to save.")
    parser.add_argument(
        "--img-size",
        type=int,
        default=32,
        help="Image height/width in pixels (grayscale square).",
    )
    parser.add_argument(
        "--syn-only",
        action="store_true",
        help="Only convert TCP SYN packets.",
    )
    args = parser.parse_args()

    pcap_path = Path(args.pcap)
    if not pcap_path.is_file():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or pcap_path.stem

    saved = 0
    for pkt_bytes in iter_pcap_packets(pcap_path):
        img_arr = packet_to_image(pkt_bytes, args.img_size, args.syn_only)
        if img_arr is None:
            continue

        img = Image.fromarray(img_arr)
        img.save(out_dir / f"{prefix}_{saved}.png")
        saved += 1

        if saved >= args.max_images:
            break

    print(f"Saved {saved} images to {out_dir}")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
