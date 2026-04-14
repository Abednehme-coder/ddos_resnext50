#!/usr/bin/env python3
"""
Blind test: pick N random images from each category (ddos, normal), run model
inference without the model knowing the labels, then report if classifications
are correct.

Usage:
  python scripts/blind_test_random.py [--data-root PATH] [--ckpt PATH] [--n-per-class N]
  python scripts/blind_test_random.py --split test --total 3000   # large blind test

The model receives only the images; true labels are used only for evaluation.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Project root for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_primary_ckpt import primary_ckpt_path, require_primary_ckpt_for_eval

import mindspore as ms
from mindspore import ops
from mindcv.models import create_model

# Match train_resnext transforms
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "model").is_dir() or (parent / "dataset").is_dir():
            return parent
    return start


def load_and_preprocess(image_path: Path, img_size: int = 224, resize_size: int = 256) -> np.ndarray:
    """Load image and apply same eval transforms as train_resnext."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    # Resize + CenterCrop (match train_resnext eval pipeline)
    if resize_size and resize_size != img_size:
        img = Image.fromarray(arr).resize((resize_size, resize_size), Image.BILINEAR)
        arr = np.array(img)
        h, w = arr.shape[:2]
        top = (h - img_size) // 2
        left = (w - img_size) // 2
        arr = arr[top : top + img_size, left : left + img_size]
    else:
        img = Image.fromarray(arr).resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(img)

    arr = arr.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return arr


def main() -> int:
    project = find_project_root(_SCRIPT_DIR)
    parser = argparse.ArgumentParser(
        description="Blind test: random images per class, model predicts without seeing labels.",
    )
    w03 = project / "dataset" / "images_per_second_window_0p3"
    legacy = project / "dataset" / "images"
    if (w03 / "train").is_dir():
        default_data = w03
    elif (legacy / "train").is_dir():
        default_data = legacy
    else:
        default_data = project / "dataset"
        if not (default_data / "train").is_dir():
            fallback = Path.home() / "datasets" / "CICDDoS2019" / "images"
            if (fallback / "train").is_dir():
                default_data = fallback
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data,
        help="Dataset root with train/val/test, each containing ddos/ and normal/.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=primary_ckpt_path(),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=None,
        help="Number of random images per category. Overridden by --total if set.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="Total images (split equally across classes). E.g. 3000 -> 1500 per class.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Data split to sample from. Use 'test' for true blind test (default: test).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device-target",
        default="CPU",
        choices=["CPU", "GPU", "Ascend"],
        help="MindSpore device.",
    )
    args = parser.parse_args()
    require_primary_ckpt_for_eval(args.ckpt)

    n_per_class = args.n_per_class
    if args.total is not None:
        n_per_class = args.total // 2
    if n_per_class is None:
        n_per_class = 5

    split_dir = args.data_root.resolve() / args.split
    ddos_dir = split_dir / "ddos"
    normal_dir = split_dir / "normal"

    for d in (ddos_dir, normal_dir):
        if not d.is_dir():
            print(f"Error: Expected directory {d}", file=sys.stderr)
            return 2

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    ddos_files = [f for f in ddos_dir.iterdir() if f.is_file() and f.suffix.lower() in image_exts]
    normal_files = [f for f in normal_dir.iterdir() if f.is_file() and f.suffix.lower() in image_exts]

    n = min(n_per_class, len(ddos_files), len(normal_files))
    if n < 1:
        print(
            f"Error: Need at least 1 image per class. "
            f"Found ddos={len(ddos_files)}, normal={len(normal_files)}.",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.seed)
    samples: list[tuple[Path, str]] = []
    for f in rng.sample(ddos_files, n):
        samples.append((f, "ddos"))
    for f in rng.sample(normal_files, n):
        samples.append((f, "normal"))

    rng.shuffle(samples)

    class_names = ["ddos", "normal"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    verbose = len(samples) <= 50
    print("Blind test: model receives images only, labels used only for evaluation.\n")
    print(f"Split: {args.split} | Picked {len(samples)} random images ({n} per class), order shuffled.\n")

    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target)
    net = create_model(model_name="resnext50_32x4d", pretrained=False, num_classes=2)
    params = ms.load_checkpoint(str(args.ckpt.resolve()))
    ms.load_param_into_net(net, params)
    net.set_train(False)

    batch_size = args.batch_size
    correct = 0
    results: list[tuple[str, str, str, bool]] = []
    conf = np.zeros((2, 2), dtype=np.int64)

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        batch_arr = np.stack([load_and_preprocess(p) for p, _ in batch], axis=0)
        x = ms.Tensor(batch_arr, dtype=ms.float32)
        logits = net(x)
        pred_idxs = np.argmax(logits.asnumpy(), axis=1)

        for j, (path, true_class) in enumerate(batch):
            pred_idx = int(pred_idxs[j])
            pred_class = idx_to_class[pred_idx]
            true_idx = class_to_idx[true_class]
            ok = pred_class == true_class
            if ok:
                correct += 1
            conf[true_idx, pred_idx] += 1
            results.append((path.name, true_class, pred_class, ok))

        if not verbose and (i + batch_size) % 500 < batch_size:
            print(f"  Processed {min(i + batch_size, len(samples))}/{len(samples)}...")

    if verbose:
        print("-" * 60)
        print(f"{'Image':<40} {'True':<8} {'Predicted':<10} {'OK'}")
        print("-" * 60)
        for name, true_c, pred_c, ok in results:
            status = "✓" if ok else "✗"
            print(f"{name:<40} {true_c:<8} {pred_c:<10} {status}")

    print("-" * 60)
    print("Confusion Matrix (rows=true, cols=predicted):")
    print(f"                ddos  normal")
    print(f"  ddos      {conf[0,0]:6d}  {conf[0,1]:6d}")
    print(f"  normal    {conf[1,0]:6d}  {conf[1,1]:6d}")
    print("-" * 60)

    acc = correct / len(results) * 100
    tp_ddos, fn_ddos = conf[0, 0], conf[0, 1]
    fp_ddos = conf[1, 0]
    prec_ddos = tp_ddos / (tp_ddos + fp_ddos) if (tp_ddos + fp_ddos) > 0 else 0
    rec_ddos = tp_ddos / (tp_ddos + fn_ddos) if (tp_ddos + fn_ddos) > 0 else 0
    f1_ddos = 2 * prec_ddos * rec_ddos / (prec_ddos + rec_ddos) if (prec_ddos + rec_ddos) > 0 else 0
    tp_norm, fn_norm = conf[1, 1], conf[1, 0]
    fp_norm = conf[0, 1]
    prec_norm = tp_norm / (tp_norm + fp_norm) if (tp_norm + fp_norm) > 0 else 0
    rec_norm = tp_norm / (tp_norm + fn_norm) if (tp_norm + fn_norm) > 0 else 0
    f1_norm = 2 * prec_norm * rec_norm / (prec_norm + rec_norm) if (prec_norm + rec_norm) > 0 else 0
    f1_macro = (f1_ddos + f1_norm) / 2

    print(f"\nAccuracy: {correct}/{len(results)} = {acc:.1f}%")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"  DDoS  - Precision: {prec_ddos:.4f}, Recall: {rec_ddos:.4f}, F1: {f1_ddos:.4f}")
    print(f"  Normal - Precision: {prec_norm:.4f}, Recall: {rec_norm:.4f}, F1: {f1_norm:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
