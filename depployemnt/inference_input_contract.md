## Inference input contract (images)

This repo trains ResNeXt on packet-derived 224x224 images, but MindSpore model input is after transforms.

### Expected image shape
- Images saved by `scripts/pcap_to_images.py` are grayscale 224x224 PNGs.
- Training transforms replicate grayscale to 3 channels (RGB-style) before normalization.

Therefore the inference tensor should match training preprocessing:
- Start with an image (or numpy array) shaped `HWC`
- Ensure it has 3 channels
- Resize/crop to `224x224` with the same eval recipe (see below)
- Normalize with ImageNet stats
- Convert to `CHW` and float32

### Eval resize/crop recipe in this repo
File: `notebook/train_resnext.py`
- Eval/Test:
  - resize to `resize_size` (default 256)
  - center-crop to 224

If you choose to skip the resize/crop (e.g., since your converter already outputs 224x224), you must ensure it truly matches what training does for evaluation. For safety, keep the same resize+center-crop logic at inference time.

### Normalization values
- mean: `[0.485, 0.456, 0.406]`
- std: `[0.229, 0.224, 0.225]`

