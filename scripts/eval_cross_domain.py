#!/usr/bin/env python3
"""
Evaluate a checkpoint trained on one image distribution on another split (domain shift).

``--train-order-root`` must be the dataset tree the checkpoint was trained on
(so ``train/`` class order matches the saved weights). ``--eval-root`` supplies
the other domain's val/ or test/ images.

Example (0.3s focal-v2 model only — checkpoint is fixed; train vs eval roots vary):

  .venv/bin/python scripts/eval_cross_domain.py \\
    --train-order-root dataset/images_per_second_window_0p3 \\
    --eval-root dataset/images --split test
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from eval_primary_ckpt import primary_ckpt_path

_DEFAULT_TRAIN_ORDER = _ROOT / "dataset" / "images_per_second_window_0p3"
_DEFAULT_EVAL_OTHER = _ROOT / "dataset" / "images"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-order-root",
        type=Path,
        default=_DEFAULT_TRAIN_ORDER,
        help="Dataset root whose train/ defines class index order (default: 0.3s windows).",
    )
    ap.add_argument(
        "--eval-root",
        type=Path,
        default=_DEFAULT_EVAL_OTHER,
        help="Root with val/ or test/ folders (default: legacy per-packet images for cross-domain).",
    )
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--device-target", default="GPU", choices=["CPU", "GPU", "Ascend"])
    args = ap.parse_args()

    train_order = args.train_order_root

    eval_py = _ROOT / "scripts" / "eval_test_detailed.py"
    cmd = [
        sys.executable,
        str(eval_py),
        "--ckpt",
        str(primary_ckpt_path()),
        "--data-root",
        str(train_order),
        "--eval-root",
        str(args.eval_root),
        "--split",
        args.split,
        "--device-target",
        args.device_target,
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
