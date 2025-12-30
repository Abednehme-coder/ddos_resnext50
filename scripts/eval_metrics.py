import argparse
import os
from collections import Counter

import numpy as np
import mindspore as ms
from mindspore import nn, ops
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

# MindSpore 1.8 lacks nn.SiLU; add a minimal implementation before importing mindcv.
if not hasattr(nn, "SiLU"):
    class SiLU(nn.Cell):
        def construct(self, x):
            return ops.Sigmoid()(x) * x
    nn.SiLU = SiLU  # type: ignore

from mindcv.models import create_model


def make_dataset(data_root, split, batch_size, shuffle=False):
    path = os.path.join(data_root, split)
    base_ds = ds.ImageFolderDataset(path, shuffle=shuffle)
    class_to_idx = base_ds.class_indexing
    if not class_to_idx:
        raise RuntimeError(f"No classes found at {path}. Check dataset path and contents.")

    def to_rgb_np(img):
        # img: HWC numpy array
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        return img

    transforms = [
        vision.Decode(),
        to_rgb_np,
        vision.Resize((224, 224)),
        vision.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_hwc=True,
        ),
        vision.HWC2CHW(),
    ]

    dataset = base_ds.map(operations=transforms, input_columns="image")
    dataset = dataset.batch(batch_size)
    return dataset, "label", class_to_idx  # label column name


def build_model(num_classes, ckpt_path, lr=1e-3):
    net = create_model(
        model_name="resnext50_32x4d",
        pretrained=False,
        num_classes=num_classes,
    )
    params = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(net, params)
    net.set_train(False)
    # Build a minimal Model just for predict/eval
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    model = ms.train.Model(net, loss_fn=loss, optimizer=optimizer, metrics={"acc"})
    return model


def compute_metrics(labels, preds, num_classes):
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for y, p in zip(labels, preds):
        if y == p:
            tp[p] += 1
        else:
            fp[p] += 1
            fn[y] += 1

    def safe_div(a, b):
        return a / b if b else 0.0

    metrics = []
    for c in range(num_classes):
        precision = safe_div(tp[c], tp[c] + fp[c])
        recall = safe_div(tp[c], tp[c] + fn[c])
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        metrics.append((precision, recall, f1, tp[c], fp[c], fn[c]))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Eval precision/recall/F1 on ImageFolder dataset.")
    parser.add_argument("--data-root", default="/workspace/ddos_data/images_balanced", help="Dataset root with train/val/test.")
    parser.add_argument("--split", default="test", help="Split to evaluate (train/val/test).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--device-target", default="GPU")
    args = parser.parse_args()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)

    dataset, label_col, class_to_idx = make_dataset(args.data_root, args.split, args.batch_size, shuffle=False)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    model = build_model(num_classes, args.ckpt)

    all_labels = []
    all_preds = []
    for batch in dataset.create_dict_iterator(output_numpy=True):
        images = batch["image"]
        labels = batch[label_col]
        logits = model.predict(ms.Tensor(images)).asnumpy()
        preds = logits.argmax(axis=1)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())

    metrics = compute_metrics(all_labels, all_preds, num_classes)

    print(f"Split: {args.split}, samples: {len(all_labels)}")
    for idx, (prec, rec, f1, tp, fp, fn) in enumerate(metrics):
        cname = idx_to_class.get(idx, str(idx))
        print(f"{cname}: precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} tp={tp} fp={fp} fn={fn}")


if __name__ == "__main__":
    main()
