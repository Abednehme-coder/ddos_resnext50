# Cross-domain evaluation (GPU)

**Date:** 2026-04-03  
**Git:** `96b82d4cf152053175950a346bdd204b6e109ebf`  
**Device:** NVIDIA GeForce RTX 3070, MindSpore 2.6.0, `scripts/eval_cross_domain.py --device-target GPU`

**Which model is “current” 0.3 s?** Offline metrics and the primary checkpoint are
`model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt` on
`dataset/images_per_second_window_0p3/`. That is **not** the same file as
`model/resnext50_32x4d_best.ckpt` at repo root (different MD5s): the latter is
the **legacy per-packet** run used only for cross-direction **B** below. ECS
should receive the 0.3 s weights via `depployemnt/deploy_ckpt_to_ecs.sh` into
`.../model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt` (same path as `run_pipeline.py` default).

Class indices follow **alphabetical** folder order under the training root (`ddos` = class 0, `normal` = class 1); verified identical for both dataset roots.

## A — 0.3 s trained checkpoint on **per-packet** test split

- **Checkpoint:** `model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt`
- **Train order (labels):** `dataset/images_per_second_window_0p3/train`
- **Eval images:** `dataset/images/test`
- **Samples:** 19300  
- **Overall accuracy:** 0.987565  
- **Macro F1:** 0.496872  
- **Weighted F1:** 0.981386  
- **Balanced accuracy:** 0.500000  

Confusion matrix [rows=true, cols=pred]:

|        | pred ddos | pred normal |
|--------|----------:|------------:|
| ddos   |         0 |         240 |
| normal |         0 |       19060 |

**Interpretation:** argmax predictions **never** assign the DDoS class on this out-of-domain representation; headline accuracy is dominated by the normal majority (cf. majority baseline acc. 0.987565).

## B — **Per-packet** trained checkpoint on 0.3 s **test** split

- **Checkpoint:** `model/resnext50_32x4d_best.ckpt`
- **Train order:** `dataset/images/train`
- **Eval images:** `dataset/images_per_second_window_0p3/test`
- **Samples:** 1845  
- **Overall accuracy:** 0.898645  
- **Macro F1:** 0.473309  
- **Weighted F1:** 0.877353  
- **Balanced accuracy:** 0.484795  

Confusion matrix [rows=true, cols=pred]:

|        | pred ddos | pred normal |
|--------|----------:|------------:|
| ddos   |         0 |         135 |
| normal |        52 |        1658 |

**Interpretation:** DDoS recall is **zero** under argmax; some normal windows are misclassified as DDoS (false positives).

## Reproduce

```bash
cd /path/to/huawei
export CUDA_HOME=/usr/local/cuda-11.6  # adjust
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:..."

.venv/bin/python scripts/eval_cross_domain.py \
  --ckpt model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt \
  --train-order-root dataset/images_per_second_window_0p3 \
  --eval-root dataset/images --split test --device-target GPU

.venv/bin/python scripts/eval_cross_domain.py \
  --ckpt model/resnext50_32x4d_best.ckpt \
  --train-order-root dataset/images \
  --eval-root dataset/images_per_second_window_0p3 --split test --device-target GPU
```
