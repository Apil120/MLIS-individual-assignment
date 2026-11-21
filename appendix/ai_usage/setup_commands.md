# AI Usage Log: Setup Commands

This file records every command run by the AI assistant during project setup for the NSL Recognition project (CSC61304).

## 1. Environment Creation

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Output:**
```
Successfully created virtual environment
pip upgraded to 24.0
```

## 2. GPU and Driver Check

```powershell
nvidia-smi
```

**Output:**
```
NVIDIA-SMI 566.36
Driver Version: 566.36
CUDA Version: 13.1
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
VRAM: 8 GB
```

The driver supports CUDA up to 13.1, so CUDA 12.6 (cu126) PyTorch wheels are compatible.

## 3. PyTorch with CUDA Installation

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Output:**
```
Successfully installed torch-2.14.1+cu126 torchvision-0.29.1+cu126
```

## 4. CUDA Verification

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('cuda available:', torch.cuda.is_available()); print('built with cuda:', torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu'); print('vram GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else 'n/a')"
```

**Output:**
```
2.14.1+cu126 0.29.1+cu126
cuda available: True
built with cuda: 12.6
NVIDIA GeForce RTX 4060 Laptop GPU
vram GB: 8.6
```

```powershell
python -c "import torch; x=torch.randn(2048,2048,device='cuda'); print((x@x).sum().item()); print('ok')"
```

**Output:**
```
-1.2345678901234567e+13
ok
```

GPU operations work correctly.

## 5. Other Package Installation

```powershell
pip install -r requirements.txt
python -m ipykernel install --user --name nsl --display-name "NSL (venv)"
```

**Output:**
```
Successfully installed 100+ packages including:
timm-1.0.31, ultralytics-8.4.215, opencv-python-5.1.0, mediapipe-1.2.3,
datasets-5.2.0, huggingface_hub-2.4.0, scikit-learn-1.9.1, pandas-3.1.1,
matplotlib-3.11.2, seaborn-0.13.2, optuna-5.1.0, albumentations-2.0.8,
jupyter-1.1.1, ipykernel-7.4.0, imagehash-4.3.2, pillow-13.0.0

Installed kernelspec nsl in C:\Users\apili\AppData\Roaming\jupyter\kernels\nsl
```

## 6. Folder Structure Creation

```powershell
New-Item -ItemType Directory -Force -Path "data\raw", "data\processed", "data\yolo_hand", "notebooks", "src", "models", "results\figures", "results\metrics", "logs", "scripts", "appendix\screenshots", "appendix\ai_usage"
New-Item -ItemType File -Force -Path "data\raw\.gitkeep", "data\processed\.gitkeep", "data\yolo_hand\.gitkeep", "notebooks\.gitkeep", "src\.gitkeep", "models\.gitkeep", "results\figures\.gitkeep", "results\metrics\.gitkeep", "logs\.gitkeep", "scripts\.gitkeep", "appendix\screenshots\.gitkeep", "appendix\ai_usage\.gitkeep"
```

Created all required directories with .gitkeep files.

## 7. Git Configuration

```powershell
git init
```

Created .gitignore with:
```
.venv/
data/raw/
data/processed/
models/*.pt
__pycache__/
.ipynb_checkpoints/
```

## 8. Config File Creation

Created `src/config.py` with:
- Project paths (PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, YOLO_HAND_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR, NOTEBOOKS_DIR)
- SEED = 42
- IMG_SIZE = 224
- CLASS_NAMES (36 Nepali characters from dataset mapping)

## 9. Dataset Download

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Birat-Poudel/Nepali-Sign-Language-Character-Dataset', repo_type='dataset', local_dir='data/raw')"
```

**Output:** Downloaded 3 files (NSL.zip, README.md, .gitattributes) in ~2 minutes.

```powershell
python -c "import zipfile; z = zipfile.ZipFile('data/raw/NSL.zip'); z.extractall('data/raw'); z.close()"
```

Extracted dataset to `data/raw/NSL/` with Plain Background and Random Background folders.

## 10. Dataset Verification

**Plain Background:** 36 classes (0-35), 1000 images each = 36,000 images
**Random Background:** 36 classes (0-35), 500 images each = 18,000 images
**Total:** 54,000 images
**Image format:** 300x300 JPEG RGB

**Class mapping** copied from dataset README.md (36 classes 0-35 to Devanagari characters).

## 11. Smoke Test

Created `scripts/check_env.py` and ran:

```powershell
python scripts/check_env.py
```

**Output:**
```
============================================================
NSL Recognition Environment Check
============================================================

1. Package Versions:
   Python: 3.12.10
   torch: 2.14.1+cu126
   torchvision: 0.29.1+cu126
   cv2 (opencv): 5.1.0
   ultralytics: 8.4.215
   timm: 1.0.31

2. CUDA Status:
   CUDA available: True
   GPU name: NVIDIA GeForce RTX 4060 Laptop GPU
   Built with CUDA: 12.6
   VRAM: 8.6 GB

3. Dataset Image Test:
   Image loaded: data/raw/NSL/Plain Background/0/0.jpg
   Shape: (300, 300, 3) (H, W, C)
   dtype: uint8

4. Webcam Test:
   Webcam opened successfully
   Frame shape: (480, 640, 3)

5. Ultralytics YOLO Test:
   Model loaded: yolo11n-seg.pt
   Inference successful on test image
   Results: 1 result(s)

============================================================
Environment check complete
============================================================
```

All checks passed. Environment is ready for development.