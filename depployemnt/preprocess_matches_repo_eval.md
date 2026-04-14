## Preprocessing matches repo eval (important)

Your model was trained/evaluated with MindSpore dataset transforms defined in `notebook/train_resnext.py`.

For deployment, match the **eval/test** preprocessing path (not the train-time augmentation):
1. Input image is grayscale 224x224, produced by `scripts/pcap_to_images.py`.
2. Training eval transforms effectively:
   - replicate grayscale to 3 channels (RGB-style)
   - resize to `resize_size` (default 256)
   - center-crop to 224
   - convert to float in [0,1]
   - ImageNet normalize:
     - mean `[0.485, 0.456, 0.406]`
     - std  `[0.229, 0.224, 0.225]`
   - convert HWC -> CHW

Repo reference implementation for this eval-like preprocessing exists in:
- `scripts/blind_test_random.py` (`load_and_preprocess()`), which:
  - converts to RGB
  - resize to `resize_size` and center-crop to `img_size`
  - normalizes with the same ImageNet mean/std
  - outputs `CHW` float32

For maximum compatibility, reuse the same preprocessing logic (or the same constants + same resize/center-crop behavior).

