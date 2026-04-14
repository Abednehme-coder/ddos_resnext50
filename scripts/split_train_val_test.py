#!/usr/bin/env python3
"""Move files from .../train/{class}/ into val/ and test/ (stratified, same seed per class).

Default root is dataset/images_per_second_window_0p3 (0.3s window images).
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"}


def list_images(d: Path) -> list[Path]:
    return sorted(
        p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "images_per_second_window_0p3",
        help="Directory containing train/, val/, test/",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train_root = args.images_root / "train"
    val_root = args.images_root / "val"
    test_root = args.images_root / "test"
    if not train_root.is_dir():
        raise SystemExit(f"Missing {train_root}")

    classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"No class folders in {train_root}")

    for i, class_name in enumerate(classes):
        src_dir = train_root / class_name
        files = list_images(src_dir)
        if not files:
            print(f"skip empty: {src_dir}")
            continue

        rng = random.Random(args.seed + i)
        rng.shuffle(files)

        n = len(files)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        n_test = n - n_train - n_val
        assert n_train + n_val + n_test == n

        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        val_dst = val_root / class_name
        test_dst = test_root / class_name
        if not args.dry_run:
            val_dst.mkdir(parents=True, exist_ok=True)
            test_dst.mkdir(parents=True, exist_ok=True)

        for subset, paths, dst in (
            ("val", val_files, val_dst),
            ("test", test_files, test_dst),
        ):
            for p in paths:
                target = dst / p.name
                if args.dry_run:
                    print(f"would move {p} -> {target}")
                else:
                    shutil.move(str(p), str(target))

        print(
            f"{class_name}: total={n} -> train={len(train_files)} val={len(val_files)} test={len(test_files)}"
        )


if __name__ == "__main__":
    main()
