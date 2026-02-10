import argparse
import os
from pathlib import Path
from typing import Optional
import numpy as np
import mindspore as ms
from mindspore import nn, ops
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.train import Model
try:
    from mindspore.train.metrics import Metric
except Exception:  # pragma: no cover
    try:
        from mindspore.train.metrics.metric import Metric  # type: ignore
    except Exception:  # pragma: no cover
        Metric = object  # type: ignore
from mindspore.train.callback import (
    Callback,
    LossMonitor,
    TimeMonitor,
    ModelCheckpoint,
    CheckpointConfig,
)
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint

# MindSpore 1.8 lacks nn.SiLU; add a minimal implementation for mindcv.
if not hasattr(nn, "SiLU"):
    class SiLU(nn.Cell):
        def construct(self, x):
            return ops.Sigmoid()(x) * x
    nn.SiLU = SiLU  # type: ignore

from mindcv.models import create_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "dataset").is_dir() or (parent / "model").is_dir() or (parent / "notebook").is_dir():
            return parent
    return start


def build_class_indexing(train_dir: Path) -> dict:
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise FileNotFoundError(f"No class subfolders found in: {train_dir}")
    return {name: idx for idx, name in enumerate(class_names)}


def count_images_per_class(train_dir: Path, class_indexing: dict) -> np.ndarray:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    counts = np.zeros(len(class_indexing), dtype=np.int64)
    for class_name, class_idx in class_indexing.items():
        class_path = train_dir / class_name
        if not class_path.is_dir():
            continue
        counts[class_idx] = sum(
            1
            for p in class_path.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        )
    return counts


def compute_class_weights(class_counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(class_counts, dtype=np.float32)
    counts = np.maximum(counts, 1.0)
    total = float(np.sum(counts))
    weights = total / (len(counts) * counts)
    weights = weights / float(np.mean(weights))
    return weights.astype(np.float32)


def make_transforms(split: str, img_size: int, resize_size: Optional[int], augment: bool):
    def to_rgb_np(img):
        # img: HWC numpy array
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        return img

    ops_list = [
        vision.Decode(),
        to_rgb_np,  # ensure 3 channels
    ]

    if split == "train" and augment:
        if hasattr(vision, "RandomResizedCrop"):
            ops_list.append(vision.RandomResizedCrop(img_size, scale=(0.8, 1.0)))
        else:
            ops_list.append(vision.Resize((img_size, img_size)))
        if hasattr(vision, "RandomHorizontalFlip"):
            ops_list.append(vision.RandomHorizontalFlip(prob=0.5))
    else:
        if resize_size is not None and resize_size != img_size:
            ops_list.append(vision.Resize((resize_size, resize_size)))
            if hasattr(vision, "CenterCrop"):
                ops_list.append(vision.CenterCrop(img_size))
            else:
                ops_list.append(vision.Resize((img_size, img_size)))
        else:
            ops_list += [vision.Resize((img_size, img_size))]

    # IMPORTANT: rescale to [0,1] before ImageNet normalization.
    ops_list += [
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, is_hwc=True),
        vision.HWC2CHW(),
    ]
    return ops_list


def make_dataset(
    data_root: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    class_indexing: Optional[dict],
    img_size: int,
    resize_size: Optional[int],
    num_workers: int,
    augment: bool,
):
    split_dir = Path(data_root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    try:
        dataset = ds.ImageFolderDataset(str(split_dir), shuffle=shuffle, class_indexing=class_indexing)
    except TypeError:
        dataset = ds.ImageFolderDataset(str(split_dir), shuffle=shuffle)

    dataset = dataset.map(
        operations=make_transforms(split, img_size, resize_size, augment),
        input_columns="image",
        num_parallel_workers=num_workers,
    )
    dataset = dataset.map(
        operations=transforms.TypeCast(ms.int32),
        input_columns="label",
        num_parallel_workers=num_workers,
    )
    dataset = dataset.batch(batch_size, drop_remainder=(split == "train"))
    return dataset

class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.clear()

    def clear(self):
        self.correct = 0
        self.total = 0

    def update(self, *inputs):
        y_pred, y_true = inputs
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]
        if hasattr(y_pred, "asnumpy"):
            y_pred = y_pred.asnumpy()
        if hasattr(y_true, "asnumpy"):
            y_true = y_true.asnumpy()
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        self.correct += int(np.sum(y_pred == y_true))
        self.total += int(y_true.size)

    def eval(self):
        return float(self.correct / self.total) if self.total else 0.0


class F1Score(Metric):
    def __init__(self, num_classes: int, average: str = "macro", eps: float = 1e-12):
        super().__init__()
        self.num_classes = int(num_classes)
        self.average = average
        self.eps = float(eps)
        self.clear()

    def clear(self):
        self.tp = np.zeros(self.num_classes, dtype=np.int64)
        self.fp = np.zeros(self.num_classes, dtype=np.int64)
        self.fn = np.zeros(self.num_classes, dtype=np.int64)

    def update(self, *inputs):
        y_pred, y_true = inputs
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]
        if hasattr(y_pred, "asnumpy"):
            y_pred = y_pred.asnumpy()
        if hasattr(y_true, "asnumpy"):
            y_true = y_true.asnumpy()

        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

        for c in range(self.num_classes):
            pred_c = y_pred == c
            true_c = y_true == c
            self.tp[c] += int(np.sum(pred_c & true_c))
            self.fp[c] += int(np.sum(pred_c & ~true_c))
            self.fn[c] += int(np.sum(~pred_c & true_c))

    def eval(self):
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)

        if self.average == "macro":
            return float(np.mean(f1))
        if self.average == "weighted":
            support = self.tp + self.fn
            total = float(np.sum(support))
            if total == 0.0:
                return 0.0
            return float(np.sum(f1 * (support / total)))

        raise ValueError(f"Unsupported average: {self.average}")


class WeightedCrossEntropyLoss(nn.Cell):
    def __init__(self, class_weights: Optional[ms.Tensor]):
        super().__init__()
        self.class_weights = class_weights
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        self.cast = ops.Cast()
        self.gather = ops.Gather()
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()
        self.eps = 1e-12

    def construct(self, logits, labels):
        labels = self.cast(labels, ms.int32)
        per_example = self.ce(logits, labels)
        if self.class_weights is None:
            return self.reduce_mean(per_example)
        w = self.gather(self.class_weights, labels, 0)
        weighted = per_example * w
        return self.reduce_sum(weighted) / (self.reduce_sum(w) + self.eps)


def load_ckpt(net, ckpt_path: str, strict: bool):
    params = load_checkpoint(ckpt_path)
    if strict:
        load_param_into_net(net, params)
        print(f"Loaded checkpoint (strict): {ckpt_path}")
        return

    net_params = net.parameters_dict()
    filtered = {}
    skipped = 0

    def candidate_names(param_name: str):
        yield param_name
        while "." in param_name:
            param_name = param_name.split(".", 1)[1]
            yield param_name

    for name, value in params.items():
        loaded = False
        for cand in candidate_names(name):
            if cand in net_params and getattr(net_params[cand], "shape", None) == getattr(value, "shape", None):
                filtered[cand] = value
                loaded = True
                break
        if not loaded:
            skipped += 1
    load_param_into_net(net, filtered)
    print(f"Loaded checkpoint (non-strict): {ckpt_path} (loaded={len(filtered)}, skipped={skipped})")


class EvalAndSaveBest(Callback):
    def __init__(
        self,
        model: Model,
        network,
        eval_dataset,
        metric_name: str,
        ckpt_dir: str,
        prefix: str,
        patience: Optional[int],
        dataset_sink_mode: bool,
    ):
        super().__init__()
        self.model = model
        self.network = network
        self.eval_dataset = eval_dataset
        self.metric_name = metric_name
        self.ckpt_dir = ckpt_dir
        self.prefix = prefix
        self.patience = patience
        self.dataset_sink_mode = dataset_sink_mode
        self.best_score = -1.0
        self.wait = 0
        self.best_ckpt_path = os.path.join(self.ckpt_dir, f"{self.prefix}_best.ckpt")

    def epoch_end(self, run_context):
        results = self.model.eval(self.eval_dataset, dataset_sink_mode=self.dataset_sink_mode)
        score = results.get(self.metric_name)
        if score is None:
            print(f"[val] {results}")
            return

        score = float(score)
        improved = score > self.best_score
        print(f"[val] {results} | best_{self.metric_name}={self.best_score:.6f}")

        if improved:
            self.best_score = score
            self.wait = 0
            os.makedirs(self.ckpt_dir, exist_ok=True)
            save_checkpoint(self.network, self.best_ckpt_path)
            print(f"Saved best checkpoint: {self.best_ckpt_path}")
            return

        if self.patience is None:
            return

        self.wait += 1
        if self.wait >= self.patience:
            print(f"Early stopping (patience={self.patience})")
            run_context.request_stop()


def build_model(
    model_name: str,
    num_classes: int,
    lr: float,
    weight_decay: float,
    class_weights: Optional[ms.Tensor],
    ckpt_path: Optional[str],
    strict_load: bool,
):
    net = create_model(
        model_name=model_name,
        pretrained=False,  # offline
        num_classes=num_classes
    )

    if ckpt_path:
        load_ckpt(net, ckpt_path, strict=strict_load)

    loss = WeightedCrossEntropyLoss(class_weights) if class_weights is not None else nn.SoftmaxCrossEntropyWithLogits(
        sparse=True, reduction="mean"
    )

    if hasattr(nn, "AdamWeightDecay"):
        optimizer = nn.AdamWeightDecay(net.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
    else:
        optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)

    model = Model(
        net,
        loss_fn=loss,
        optimizer=optimizer,
        metrics={
            "acc": Accuracy(),
            "f1": F1Score(num_classes=num_classes, average="macro"),
        },
    )
    return model, net


def parse_args():
    project_root = find_project_root(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(description="Train/Eval ResNeXt50 on packet images.")
    parser.add_argument(
        "--data-root",
        default=str(project_root / "dataset"),
        help="Dataset root with train/val/test (ImageFolder layout).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "model"),
        help="Directory to save checkpoints (matches the /model folder in your project layout).",
    )
    parser.add_argument("--model-name", default="resnext50_32x4d")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=None, help="Auto-inferred from train/ if omitted.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256, help="Eval resize before center crop (set to 0 to disable).")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-augment", action="store_true", help="Disable train-time augmentation.")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable auto class-weighting.")
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint to load before train/eval.")
    parser.add_argument("--strict-load", action="store_true", help="Strict checkpoint loading (will error on shape mismatch).")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate.")
    parser.add_argument("--mode", choices=["GRAPH", "PYNATIVE"], default="GRAPH")
    parser.add_argument(
        "--device-target",
        default="CPU",
        choices=["CPU", "GPU", "Ascend"],
        help="MindSpore device target. Use GPU/Ascend only if your environment supports it.",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-sink", action="store_true", help="Enable dataset sink mode (often faster on Ascend).")
    parser.add_argument("--early-stop", type=int, default=0, help="Early stop patience (0 disables). Uses val F1.")
    return parser.parse_args()


def main():
    args = parse_args()
    mode = ms.GRAPH_MODE if args.mode == "GRAPH" else ms.PYNATIVE_MODE
    ms.set_context(mode=mode, device_target=args.device_target, device_id=args.device_id)
    ms.set_seed(args.seed)
    if hasattr(ds, "config") and hasattr(ds.config, "set_seed"):
        ds.config.set_seed(args.seed)

    train_dir = Path(args.data_root) / "train"
    class_indexing = build_class_indexing(train_dir)
    num_classes = len(class_indexing) if args.num_classes is None else int(args.num_classes)
    if num_classes != len(class_indexing):
        print(f"WARNING: --num-classes={num_classes} but found {len(class_indexing)} classes in {train_dir}")

    resize_size = None if args.resize_size == 0 else int(args.resize_size)
    augment = not args.no_augment

    train_ds = make_dataset(
        args.data_root,
        "train",
        args.batch_size,
        True,
        class_indexing,
        args.img_size,
        resize_size,
        args.num_workers,
        augment,
    )
    val_ds = make_dataset(
        args.data_root,
        "val",
        args.batch_size,
        False,
        class_indexing,
        args.img_size,
        resize_size,
        args.num_workers,
        False,
    )
    test_ds = make_dataset(
        args.data_root,
        "test",
        args.batch_size,
        False,
        class_indexing,
        args.img_size,
        resize_size,
        args.num_workers,
        False,
    )

    class_weights = None
    if not args.no_class_weights:
        class_counts = count_images_per_class(train_dir, class_indexing)
        class_weights = ms.Tensor(compute_class_weights(class_counts), dtype=ms.float32)
        print(f"Class counts: {class_counts.tolist()} | class weights: {class_weights.asnumpy().tolist()}")

    model, net = build_model(
        model_name=args.model_name,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        ckpt_path=args.ckpt,
        strict_load=args.strict_load,
    )

    if not args.eval_only:
        print("\nStarting training...\n")
        os.makedirs(args.output_dir, exist_ok=True)
        steps_per_epoch = train_ds.get_dataset_size()
        if steps_per_epoch == 0:
            raise RuntimeError("Training dataset is empty. Check your dataset/train folder.")

        ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=5)
        ckpt_cb = ModelCheckpoint(prefix=args.model_name, directory=args.output_dir, config=ckpt_config)
        eval_cb = EvalAndSaveBest(
            model=model,
            network=net,
            eval_dataset=val_ds,
            metric_name="f1",
            ckpt_dir=args.output_dir,
            prefix=args.model_name,
            patience=(None if args.early_stop <= 0 else int(args.early_stop)),
            dataset_sink_mode=args.dataset_sink,
        )

        model.train(
            args.epochs,
            train_ds,
            callbacks=[LossMonitor(), TimeMonitor(), ckpt_cb, eval_cb],
            dataset_sink_mode=args.dataset_sink,
        )

    print("\nEvaluating on test set...\n")
    best_ckpt_path = os.path.join(args.output_dir, f"{args.model_name}_best.ckpt")
    if os.path.isfile(best_ckpt_path):
        load_ckpt(net, best_ckpt_path, strict=True)
    metrics = model.eval(test_ds, dataset_sink_mode=args.dataset_sink)
    print(metrics)


if __name__ == "__main__":
    main()
