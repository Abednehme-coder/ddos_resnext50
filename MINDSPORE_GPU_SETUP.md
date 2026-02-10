# MindSpore GPU – final setup steps

Your `.bashrc` is updated so `libcuda.so` (in `/lib/x86_64-linux-gnu`) and `libcudnn.so` (in `/usr/lib/x86_64-linux-gnu`) are found.

## Official doc summary

- **CUDA:** MindSpore GPU supports **CUDA 11.1 and 11.6**. Driver: CUDA 11.6 needs ≥ 510.39.01 (`nvidia-smi`).
- **cuDNN:** For CUDA 11.6 use cuDNN v8.5.x; copy headers/libs into your CUDA path (e.g. `/usr/local/cuda-11.6`).
- **Official install (nightly):**  
  `pip install mindspore-dev -i https://repo.huaweicloud.com/repository/pypi/simple/`  
  The doc says nightly "will configure automatically according to the version of CUDA installed in your environment." In practice, the latest nightly on some platforms is built for **CUDA 12** only, so you may see `libcublas.so.12` on a CUDA 11.6 system — see section 8.
- **DT_RPATH:** If `/usr/local/cuda` exists, MindSpore uses DT_RPATH and **LD_LIBRARY_PATH can be ignored**. So either have `/usr/local/cuda` point to your 11.6 install, or use a MindSpore wheel built for CUDA 11.x.
- **Env for verification:**  
  `export PATH=/usr/local/cuda-11.6/bin:$PATH`  
  `export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH`  
  `export CUDA_HOME=/usr/local/cuda-11.6`

## 1. Install cuRAND (required for MindSpore GPU)

Run in a terminal:

```bash
sudo apt-get install -y libcurand-11-6
```

## 2. Install cuSOLVER (required for MindSpore GPU)

```bash
sudo apt-get install -y libcusolver-11-6
```

## 3. If MindSpore still asks for libcurand.so.10

CUDA 11.6 may install `libcurand.so.11`. If the GPU check still fails on `libcurand.so.10`, add a symlink:

```bash
sudo ln -sf /usr/local/cuda-11.6/lib64/libcurand.so.11 /usr/local/cuda-11.6/lib64/libcurand.so.10
```

(If the file is named only `libcurand.so`, use that as the target instead of `libcurand.so.11`.)

## 4. Install remaining CUDA 11.6 libs (if more .so are reported missing)

Install in one go to avoid repeated errors:

```bash
sudo apt-get install -y libcusparse-11-6 cuda-nvrtc-11-6 2>/dev/null; true
```

If a specific library is still missing (e.g. `libcusparse.so.11`), install it:

```bash
sudo apt-get install -y libcusparse-11-6
```

## 5. If run_check still says "libcuda.so" or "libcudnn.so" not found

MindSpore may look for these under the CUDA path first. Add symlinks so they are found there:

```bash
sudo ln -sf /lib/x86_64-linux-gnu/libcuda.so /usr/local/cuda-11.6/lib64/libcuda.so
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/local/cuda-11.6/lib64/libcudnn.so
```

Then run the test again (with `LD_LIBRARY_PATH` set as in section 6).

## 6. Set CUDA_HOME (required for GPU plugin)

MindSpore uses `CUDA_HOME` to select and load the correct GPU plugin (e.g. 11.6). Without it, only CPU is registered and `set_context(device_target='GPU')` fails.

Add to `~/.bashrc` (or set in the shell before running):

```bash
export CUDA_HOME=/usr/local/cuda-11.6
```

## 7. Test

Open a **new** terminal (so `.bashrc` is loaded) or run `source ~/.bashrc`, then in the project with venv active:

```bash
cd ~/Documents/huawei
source .venv/bin/activate
# As per official doc: PATH, LD_LIBRARY_PATH, CUDA_HOME
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.6
# Quick check
python -c "import mindspore as ms; ms.set_context(device_target='GPU'); print(ms.get_context('device_target'))"
# Official verification (optional)
python -c "import mindspore; mindspore.set_device(device_target='GPU'); mindspore.run_check()"
```

You should see `GPU` and then "MindSpore has been installed on platform [GPU] successfully!". If another `.so` is reported missing, install the matching package (e.g. `libcusparse-11-6`) from the same CUDA 11.6 repo.

## 8. If you see "libcublas.so.12" or "libcublasLt.so.12" (CUDA 12 libs)

The official site recommends `pip install mindspore-dev` and says nightly supports CUDA 11.1/11.6 and "configures automatically." On some platforms the **latest** nightly is built for CUDA 12 only, so you get `.so.12` on a CUDA 11.6 machine. Fix: use the **stable** MindSpore wheel (built for CUDA 11) instead of the current nightly:

```bash
cd ~/Documents/huawei
source .venv/bin/activate
pip uninstall mindspore mindspore-dev -y
# Stable 2.6.0 from official repo (~800MB download; run in terminal so it doesn't time out)
pip install --default-timeout=600 mindspore==2.6.0 -i https://repo.mindspore.cn/pypi/simple --trusted-host repo.mindspore.cn --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple/
pip install "numpy>=1.23,<2" "scipy>=1.10,<1.14"
pip install -r requirements.txt
```

Then run the test in section 7 again. If you still see `.so.12` errors, use the [MindSpore versions page](https://www.mindspore.cn/versions/en/) to select your OS, Python, and **CUDA 11.6** and follow the given pip command for the cu116 wheel.

**DT_RPATH:** If you have `/usr/local/cuda` (symlink or directory), it is used by MindSpore before `LD_LIBRARY_PATH`. To use CUDA 11.6 with a nightly that might otherwise load CUDA 12 libs, ensure `/usr/local/cuda` points to 11.6:  
`sudo ln -sfn /usr/local/cuda-11.6 /usr/local/cuda`
