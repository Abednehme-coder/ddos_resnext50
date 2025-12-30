import argparse
import os
import numpy as np
import mindspore as ms
from mindspore import nn, ops
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore import load_checkpoint, load_param_into_net

# MindSpore 1.8 lacks nn.SiLU; add a minimal implementation for mindcv.
if not hasattr(nn, "SiLU"):
    class SiLU(nn.Cell):
        def construct(self, x):
            return ops.Sigmoid()(x) * x
    nn.SiLU = SiLU  # type: ignore

from mindcv.models import create_model

def make_dataset(data_root, split, batch_size, shuffle):
    path = os.path.join(data_root, split)
    dataset = ds.ImageFolderDataset(path, shuffle=shuffle)

    def to_rgb_np(img):
        # img: HWC numpy array
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        return img

    transforms = [
        vision.Decode(),
        to_rgb_np,  # ensure 3 channels
        vision.Resize((224, 224)),
        vision.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_hwc=True
        ),
        vision.HWC2CHW()
    ]

    dataset = dataset.map(operations=transforms, input_columns="image")
    dataset = dataset.batch(batch_size)
    return dataset

def build_model(num_classes, lr, ckpt_path=None):
    net = create_model(
        model_name="resnext50_32x4d",
        pretrained=False,  # offline
        num_classes=num_classes
    )

    if ckpt_path:
        params = load_checkpoint(ckpt_path)
        load_param_into_net(net, params)
        print(f"Loaded checkpoint: {ckpt_path}")

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)

    model = Model(
        net,
        loss_fn=loss,
        optimizer=optimizer,
        metrics={"acc"}
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train/Eval ResNeXt50 on packet images.")
    parser.add_argument("--data-root", default="/workspace/ddos_data/images_balanced", help="Dataset root with train/val/test.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint to load before train/eval.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate.")
    parser.add_argument("--device-target", default="GPU", help="MindSpore device target (GPU/CPU).")
    return parser.parse_args()


def main():
    args = parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)

    train_ds = make_dataset(args.data_root, "train", args.batch_size, True)
    val_ds   = make_dataset(args.data_root, "val", args.batch_size, False)
    test_ds  = make_dataset(args.data_root, "test", args.batch_size, False)

    model = build_model(args.num_classes, args.lr, args.ckpt)

    if not args.eval_only:
        print("\nStarting training...\n")
        os.makedirs("./models", exist_ok=True)
        ckpt_config = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=3)
        ckpt_cb = ModelCheckpoint(prefix="resnext50", directory="./models", config=ckpt_config)

        model.train(
            args.epochs,
            train_ds,
            callbacks=[LossMonitor(), TimeMonitor(), ckpt_cb],
            dataset_sink_mode=False
        )

    print("\nEvaluating on test set...\n")
    metrics = model.eval(test_ds, dataset_sink_mode=False)
    print(metrics)


if __name__ == "__main__":
    main()
