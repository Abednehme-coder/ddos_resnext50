# huawei

ResNeXt50 training on packet images using Huawei MindSpore. Training entrypoint: `notebook/train_resnext.py`. Dataset layout: `dataset/{train,val,test}` with ImageFolder (class subfolders).

## GPU training: requirements and setup

### Prerequisites (system)

- **NVIDIA GPU** with compute capability supported by your CUDA version.
- **CUDA Toolkit** (e.g. 11.6, 11.8) and **cuDNN** installed and on `PATH` / `LD_LIBRARY_PATH`. MindSpore GPU is built for specific CUDA versions; use the install command that matches your machine.
- **Python**: 3.8–3.11 (see [MindSpore GPU install](https://www.mindspore.cn/install/en) for the exact matrix).

### 1. Confirm CUDA and Python

```bash
nvidia-smi
nvcc --version
python3 --version
```

Ensure Python is in the range supported by MindSpore (e.g. 3.9–3.11).

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
```

### 3. Install MindSpore GPU

Use the **official pip command** from [MindSpore install](https://www.mindspore.cn/install/en) (Obtaining Installation Commands) for your Version, Hardware Platform (GPU), CUDA version, OS, and Python version.

Example for **GPU + CUDA 11.6 + Linux-x86_64 + Python 3.9 (nightly)**:

```bash
pip install mindspore-dev -i https://repo.mindspore.cn/pypi/nightly/simple --trusted-host repo.mindspore.cn --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple
```

Before running, add any required environment variables from the [installation guide](https://www.mindspore.cn/install/en) (e.g. `LD_LIBRARY_PATH` for CUDA/cuDNN). For a stable release instead of nightly, select the desired version on the same page and use the generated command.

### 4. Install mindcv and other Python deps

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install mindcv numpy Pillow
```

### 5. Verify GPU and run training

Confirm MindSpore sees the GPU:

```bash
python -c "import mindspore as ms; ms.set_context(device_target='GPU'); print(ms.get_context('device_target'))"
```

Ensure a dataset exists at `dataset/train`, `dataset/val`, `dataset/test` (each with class subfolders). Run training:

```bash
python notebook/train_resnext.py --device-target GPU --data-root ./dataset --output-dir ./model
```

Use `--device-id 0` (or another ID) if you have multiple GPUs.

### Notes

- **CUDA version**: Your system CUDA (and cuDNN) should match the version you selected in the MindSpore installer (e.g. CUDA 11.6).
- **Dataset**: Images in `dataset/train/<class>/`, `dataset/val/<class>/`, `dataset/test/<class>/` (e.g. `.jpg`, `.png`). Use the scripts in `scripts/` to convert PCAPs to images if needed.
