#!/usr/bin/env python3
"""
Move images that were synced into train/ (flat) into train/<class>/ using
classified manifest(s). Use after a sync that left images in train/ instead of
train/ddos/ and train/normal/.

Usage:
  python scripts/move_images_to_class_folders.py [--images-dir dataset/images] [--dry-run]

Expects:
  - images_dir/train/ may contain loose .png files and/or class subdirs (ddos, normal).
  - images_dir/classified_manifest*.txt with lines: path\\tclass (path = .../SESSION_ID).
Image filenames are assumed to be SESSION_ID_INDEX.png (e.g. SAT-01-12-2018_0500_0.png).
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def load_manifest_prefix_to_class(
    manifest_path: Path,
    default_class: str | None = None,
) -> dict[str, str]:
    """Build mapping: path basename (session id) -> class (ddos/normal).
    Lines can be 'path\\tclass' or 'path class'. If only path (no class) and
    default_class is set, use that.
    """
    out = {}
    with open(manifest_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Support tab or space separator
            parts = line.split("\t", 1) if "\t" in line else line.split(None, 1)
            path_part = parts[0].strip()
            class_name = parts[1].strip() if len(parts) > 1 else None
            if class_name is None:
                class_name = default_class
            if class_name is None:
                continue
            key = Path(path_part).name
            out[key] = class_name
    return out


def collect_manifests(
    images_dir: Path,
    default_class: str | None = None,
) -> dict[str, str]:
    """Merge all classified_manifest*.txt in images_dir into one prefix -> class map."""
    merged = {}
    for p in sorted(images_dir.glob("classified_manifest*.txt")):
        for k, v in load_manifest_prefix_to_class(p, default_class=default_class).items():
            merged[k] = v
    return merged


def session_from_filename(name: str) -> str | None:
    """From 'SAT-01-12-2018_0500_0.png' return 'SAT-01-12-2018_0500'."""
    m = re.match(r"^(.+)_\d+\.(png|jpg|jpeg|bmp|gif|tif|tiff|webp)$", name, re.I)
    return m.group(1) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move loose images in train/ into train/<class>/ using manifest(s)."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("dataset/images"),
        help="Directory containing train/ and classified_manifest*.txt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be moved, do not move.",
    )
    parser.add_argument(
        "--default-class",
        type=str,
        default=None,
        help="For manifest lines that have path only (no class), use this class (e.g. normal).",
    )
    args = parser.parse_args()

    images_dir = args.images_dir.resolve()
    train_dir = images_dir / "train"
    if not train_dir.is_dir():
        print(f"Error: {train_dir} is not a directory.", file=sys.stderr)
        return 2

    prefix_to_class = collect_manifests(images_dir, default_class=args.default_class)
    if not prefix_to_class:
        print(
            f"Error: No classified_manifest*.txt found in {images_dir} or they are empty.",
            file=sys.stderr,
        )
        return 2

    # Only consider files directly under train/ (not inside ddos/ or normal/).
    loose = [f for f in train_dir.iterdir() if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")]
    if not loose:
        print("No loose image files in train/. Nothing to do.")
        return 0

    moved = 0
    skipped = 0
    for f in loose:
        session = session_from_filename(f.name)
        if session is None:
            skipped += 1
            continue
        class_name = prefix_to_class.get(session)
        if class_name is None:
            skipped += 1
            continue
        dest_dir = train_dir / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f.name
        if dest.exists() and dest.resolve() == f.resolve():
            continue
        if args.dry_run:
            print(f"  would move: {f.relative_to(images_dir)} -> train/{class_name}/")
        else:
            shutil.move(str(f), str(dest))
        moved += 1

    if args.dry_run:
        print(f"Dry run: would move {moved} files, skip {skipped} (no session or unknown session).")
    else:
        print(f"Moved {moved} images into train/<class>/; skipped {skipped} (no session or unknown session).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
