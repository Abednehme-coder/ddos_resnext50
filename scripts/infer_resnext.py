import argparse
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import nn, ops
from PIL import Image

# MindSpore 1.8 lacks nn.SiLU; add a minimal implementation before importing mindcv.
if not hasattr(nn, "SiLU"):
    class SiLU(nn.Cell):
        def construct(self, x):
            return ops.Sigmoid()(x) * x
    nn.SiLU = SiLU  # type: ignore

from mindcv.models import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a folder of images with ResNeXt50.")
    parser.add_argument("--images-dir", required=True, help="Directory containing images.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path (e.g., models/resnext50-8_386.ckpt).")
    parser.add_argument("--class-names", default="ddos,normal", help="Comma-separated class names in index order.")
    parser.add_argument("--device-target", default="GPU", help="MindSpore device target (GPU/CPU).")
    return parser.parse_args()


def build_model(num_classes, ckpt_path):
    net = create_model(
        model_name="resnext50_32x4d",
        pretrained=False,
        num_classes=num_classes
    )
    params = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(net, params)
    net.set_train(False)
    return net


def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # CHW
    return ms.Tensor(arr[None, ...])  # NCHW


def main():
    args = parse_args()
    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    if not class_names:
        raise ValueError("At least one class name must be provided.")

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)

    net = build_model(len(class_names), args.ckpt)
    softmax = ops.Softmax()

    img_dir = Path(args.images_dir)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}

    # Collect images recursively so we can pass a parent directory with subfolders.
    images = sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")

    for img_path in images:
        x = preprocess_image(img_path)
        logits = net(x)
        probs = softmax(logits)[0].asnumpy()
        pred_idx = int(np.argmax(probs))
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        prob = float(probs[pred_idx])
        print(f"{img_path.name}: {pred_name} ({prob:.4f})")


if __name__ == "__main__":
    main()
