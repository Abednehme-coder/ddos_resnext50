import argparse
import mindspore as ms
from mindspore import nn, ops

# MindSpore 1.8 lacks nn.SiLU; add a minimal implementation for mindcv.
if not hasattr(nn, "SiLU"):
    class SiLU(nn.Cell):
        def construct(self, x):
            return ops.Sigmoid()(x) * x
    nn.SiLU = SiLU  # type: ignore

from mindcv.models import create_model


def parse_args():
    p = argparse.ArgumentParser(description="Export ResNeXt50 model to MindIR/ONNX/AIR.")
    p.add_argument("--ckpt", required=True, help="Checkpoint path (e.g., models/resnext50-8_386.ckpt).")
    p.add_argument("--out", required=True, help="Output file path without extension (export adds suffix).")
    p.add_argument("--format", default="MINDIR", choices=["MINDIR", "AIR", "ONNX"], help="Export format.")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--device-target", default="GPU", help="Device target for export (GPU/CPU).")
    return p.parse_args()


def build_model(num_classes, ckpt_path):
    net = create_model(
        model_name="resnext50_32x4d",
        pretrained=False,
        num_classes=num_classes,
    )
    params = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(net, params)
    net.set_train(False)
    return net


def main():
    args = parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)

    net = build_model(args.num_classes, args.ckpt)

    # Dummy input matching training shape (NCHW).
    dummy_input = ms.Tensor(ms.numpy.zeros((1, 3, 224, 224)), ms.float32)

    ms.export(net, dummy_input, file_name=args.out, file_format=args.format)
    print(f"Exported to {args.out}.{args.format.lower()}")


if __name__ == "__main__":
    main()
