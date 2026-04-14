#!/usr/bin/env python3
"""
ROC/PR curves, threshold sweep, expected calibration error (ECE) on val or test.
Saves figures under research/figures/ (or --out-dir).

  .venv/bin/python scripts/eval_roc_pr_calibration.py \\
    --ckpt model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt \\
    --data-root dataset/images_per_second_window_0p3 --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "notebook") not in sys.path:
    sys.path.insert(0, str(_ROOT / "notebook"))

from eval_primary_ckpt import primary_ckpt_path, require_primary_ckpt_for_eval

import mindspore as ms

import train_resnext as tr


def softmax_np(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def binary_metrics_at_threshold(
    y_true: np.ndarray, p_ddos: np.ndarray, thr: float, ddos_idx: int
) -> tuple[float, float, float]:
    y_pos = (y_true == ddos_idx).astype(np.int64)
    pred_pos = (p_ddos >= thr).astype(np.int64)
    tp = int(np.sum((pred_pos == 1) & (y_pos == 1)))
    fp = int(np.sum((pred_pos == 1) & (y_pos == 0)))
    fn = int(np.sum((pred_pos == 0) & (y_pos == 1)))
    eps = 1e-12
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return float(prec), float(rec), float(f1)


def roc_points(y_true_bin: np.ndarray, p_ddos: np.ndarray, n_points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """y_true_bin: 1 if true class is DDoS (positive), 0 if normal."""
    thrs = np.linspace(0, 1, n_points)
    tpr_list = []
    fpr_list = []
    pos = y_true_bin == 1
    neg = ~pos
    n_pos = max(int(pos.sum()), 1)
    n_neg = max(int(neg.sum()), 1)
    for t in thrs:
        pred_pos = p_ddos >= t
        tp = np.sum(pred_pos & pos)
        fp = np.sum(pred_pos & neg)
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    return np.array(fpr_list), np.array(tpr_list)


def pr_points(y_true_bin: np.ndarray, p_ddos: np.ndarray, n_points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    thrs = np.linspace(0, 1, n_points)
    prec_list = []
    rec_list = []
    pos = y_true_bin == 1
    for t in thrs:
        pred_pos = p_ddos >= t
        tp = np.sum(pred_pos & pos)
        fp = np.sum(pred_pos & ~pos)
        fn = np.sum(~pred_pos & pos)
        eps = 1e-12
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        prec_list.append(prec)
        rec_list.append(rec)
    return np.array(rec_list), np.array(prec_list)


def ece_binary(p_ddos: np.ndarray, y_true_bin: np.ndarray, n_bins: int = 10) -> float:
    """ECE for DDoS as positive class; p_ddos is model confidence for DDoS."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(p_ddos)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p_ddos >= lo) & (p_ddos < hi if i < n_bins - 1 else p_ddos <= hi)
        if not np.any(m):
            continue
        conf = float(p_ddos[m].mean())
        acc = float(y_true_bin[m].mean())
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=_ROOT / "dataset" / "images_per_second_window_0p3")
    ap.add_argument("--ckpt", type=Path, default=primary_ckpt_path())
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "research" / "figures")
    ap.add_argument(
        "--device-target",
        default="GPU",
        choices=["CPU", "GPU"],
        help="MindSpore device (default: GPU for eval speed; set CPU for hosts without CUDA).",
    )
    args = ap.parse_args()
    require_primary_ckpt_for_eval(args.ckpt)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib", file=sys.stderr)
        return 1

    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target)
    train_dir = args.data_root / "train"
    class_indexing = tr.build_class_indexing(train_dir)
    class_names = sorted(class_indexing.keys(), key=lambda n: class_indexing[n])
    assert len(class_names) == 2
    ddos_idx = class_names.index("ddos")

    net = tr.create_model("resnext50_32x4d", pretrained=False, num_classes=2)
    tr.load_ckpt(net, str(args.ckpt), strict=True)
    net.set_train(False)

    ds = tr.make_dataset(
        str(args.data_root),
        args.split,
        32,
        False,
        class_indexing,
        224,
        256,
        4,
        False,
    )
    logits_list = []
    labels_list = []
    for batch in ds.create_dict_iterator(num_epochs=1):
        imgs = batch["image"]
        lab = batch["label"]
        logits = net(imgs)
        logits_list.append(logits.asnumpy())
        labels_list.append(lab.asnumpy())

    logits = np.concatenate(logits_list, axis=0)
    y_true = np.concatenate(labels_list, axis=0).astype(np.int64)
    probs = softmax_np(logits)
    p_ddos = probs[:, ddos_idx].astype(np.float64)
    y_ddos_pos = (y_true == ddos_idx).astype(np.int64)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ROC
    fpr, tpr = roc_points(y_ddos_pos, p_ddos)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC ({args.split}, n={len(y_true)})")
    ax.legend()
    fig.tight_layout()
    roc_path = args.out_dir / f"roc_{args.split}.pdf"
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {roc_path}")

    # PR
    rec, prec = pr_points(y_ddos_pos, p_ddos)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, label="PR")
    ax.set_xlabel("Recall (DDoS)")
    ax.set_ylabel("Precision (DDoS)")
    ax.set_title(f"Precision–recall ({args.split})")
    ax.legend()
    fig.tight_layout()
    pr_path = args.out_dir / f"pr_{args.split}.pdf"
    fig.savefig(pr_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pr_path}")

    # Threshold sweep (F1 for predicting DDoS if p_ddos >= t)
    thrs = np.linspace(0.01, 0.99, 50)
    f1s = []
    for t in thrs:
        _, _, f1 = binary_metrics_at_threshold(y_true, p_ddos, t, ddos_idx)
        f1s.append(f1)
    best_i = int(np.argmax(f1s))
    print(f"Best threshold (max F1 heuristic): {thrs[best_i]:.4f}, F1≈{f1s[best_i]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thrs, f1s)
    ax.set_xlabel(r"Threshold $\tau$ on $p_{\mathrm{ddos}}$")
    ax.set_ylabel("F1 (DDoS positive)")
    ax.set_title("Threshold sweep (argmax is τ=0.5 for binary softmax tie-break)")
    fig.tight_layout()
    sweep_path = args.out_dir / f"threshold_f1_{args.split}.pdf"
    fig.savefig(sweep_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {sweep_path}")

    ece = ece_binary(p_ddos, y_ddos_pos)
    print(f"ECE (approx., {args.split}): {ece:.4f}")

    # Confusion heatmap
    pred_cls = np.argmax(probs, axis=1)
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, pred_cls):
        cm[int(t), int(p)] += 1
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred DDoS", "pred normal"])
    ax.set_yticklabels(["true DDoS", "true normal"])
    ax.set_title(f"Confusion ({args.split})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    cm_path = args.out_dir / f"confusion_heatmap_{args.split}.pdf"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {cm_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
