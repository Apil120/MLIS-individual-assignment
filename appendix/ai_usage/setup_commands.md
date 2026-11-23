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

---

## 12. MediaPipe HandLandmarker Integration (Post-Setup Development)

### 12.1 Created MediaPipe Testing Script

```powershell
# Created testing_mediapipe.py with MediaPipe HandLandmarker
# Auto-downloads hand_landmarker.task model from Google
# Detects 21 hand landmarks with connections and bounding box
```

### 12.2 Reorganized Code Structure

```powershell
# Created utils.py - Shared utilities:
#   - HAND_CONNECTIONS (21 landmark connections)
#   - download_model_if_needed()
#   - create_detector() - MediaPipe HandLandmarker
#   - draw_hand_landmarks() - Draw landmarks + skeleton

# Updated main.py - Clean entry point:
#   - Imports from utils and classifier
#   - Real-time webcam with hand detection + NSL classification
```

### 12.3 Updated setup.md for MediaPipe Pipeline

```powershell
# Changed pipeline from YOLO → MediaPipe Hand Landmarker
# Removed ultralytics from requirements.txt
# Updated smoke test to verify MediaPipe instead of YOLO
# Updated folder structure: yolo_hand/ → mediapipe_hand/
# Updated reference plan (Section 7) for MediaPipe approach
```

### 12.4 Dataset Exploration Script

```powershell
python scripts/explore_dataset.py
```

**Output:**
```
Plain Background:  36 classes, 1000 images/class = 36,000 total
Random Background: 36 classes, 500 images/class  = 18,000 total
Grand Total: 54,000 images
Resolution: 300x300 JPEG, 3-channel BGR
```

### 12.5 Added MediaPipe Visualization to eda.ipynb

```python
# Cell: MediaPipe HandLandmarker Visualization
# - Builds HandLandmarker (RunningMode.IMAGE, num_hands=1)
# - Samples 4 plain + 4 random background images from labels.csv
# - Runs detection, draws 21 landmarks + connections + bbox
# - Displays 2x4 grid with titles: "label | background | detected/not detected"
```

### 12.6 Created CNN Classifier (classifier.py)

```python
# CNN architecture matching trained model (best_model.pth):
#   - 3 Conv layers (1→16→32→64) + MaxPool + AdaptiveAvgPool(7x7)
#   - 2 FC layers (64*7*7→512→36) + Dropout(0.5)
# 
# NSLClassifier class:
#   - Loads best_model.pth weights
#   - Preprocessing: Resize(300,300) → Grayscale → Normalize(0.5, 0.5)
#   - Predicts class_idx, Devanagari char, confidence, full probabilities
#   - Extracts hand crop from MediaPipe landmarks with padding
#
# CLASS_NAMES mapping (36 classes from dataset README):
#   0:क  1:ख  2:ग  3:घ  4:ङ  5:च  6:छ  7:ज  8:झ  9:ञ
#   10:ट 11:ठ 12:ड 13:ढ 14:ण 15:त 16:थ 17:द 18:ध 19:न
#   20:प 21:फ 22:ब 23:भ 24:म 25:य 26:र 27:ल 28:व 29:श
#   30:ष 31:स 32:ह 33:क्ष 34:त्र 35:ज्ञ
```

### 12.7 Integrated MediaPipe + CNN in main.py

```python
# Real-time pipeline:
#   1. Webcam frame → BGR to RGB
#   2. MediaPipe HandLandmarker detects landmarks
#   3. Draw landmarks (green) + connections (blue) + bbox (yellow)
#   4. Extract hand crop from landmarks
#   5. CNN predicts NSL character + confidence
#   6. Display prediction on frame (color-coded by confidence)
```

**Test Result:**
```
Predicted: Class 0 - Confidence: 98.46%
(Class 0 = 'क' - matches plain_background/0/0.jpg)
```

### 12.8 Added Model Evaluation Cell to eda.ipynb

```python
# Cell: Model Evaluation - Precision, Recall, F1-Score, Confusion Matrix
# - Loads best_model.pth, recreates exact test split (20%, stratified, seed=42)
# - Runs inference on 10,800 test images
# - Computes:
#   * Macro & Weighted Precision, Recall, F1
#   * Per-class metrics for all 36 classes
#   * Confusion Matrix (counts) - 36x36 heatmap
#   * Normalized Confusion Matrix (row-wise %) - 36x36 heatmap
#   * Per-class F1 bar chart with macro F1 reference line
#   * Classification Report as DataFrame
```