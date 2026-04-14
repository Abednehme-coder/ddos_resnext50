## Open questions before we implement (or export) for Huawei Cloud

1. Streaming window definition
   - Decision (default for now): **time-based windows**.
   - Suggested start: **2 seconds per window**.
   - Rationale: works naturally with continuous capture and gives predictable alert cadence.
   - Latency impact: ~window_duration + processing time.

2. Image cap per window
   - Decision (default for now): **cap at 200 images per view** (per window, per branch).
   - Rationale: matches `scripts/pcap_to_images.py` default `--max-images=200`, so compute bounds and input ordering behavior are closer to offline dataset generation.
   - If your window yields more than cap: take the **first N** converted images to mirror “stop after max-images” behavior.
   - If your window yields fewer: use the available images.

3. Preprocessing consistency
   - Decision (recommended): use the **eval/test preprocessing** from this repo:
     - grayscale -> RGB replication (3 channels)
     - resize to `resize_size=256`
     - center-crop to `224`
     - ImageNet mean/std normalization
   - Reference: `notebook/train_resnext.py` (`make_transforms()` eval path) and `scripts/blind_test_random.py` (`load_and_preprocess()`).

4. Model runtime artifact for Huawei Cloud
   - what artifact you currently have (ckpt / mindir / air / etc.)
   - whether target runtime is GPU or Ascend
   - export/compile steps required by the serving environment

5. Threshold tuning
   - decide decision threshold over fused `p(ddos)` using validation data
   - choose acceptable precision/recall trade-off for alerts
   - For now: you can start with **argmax-of-fused-probabilities** (same as training/eval classification logic), then calibrate a custom threshold later if you want different alert behavior.

