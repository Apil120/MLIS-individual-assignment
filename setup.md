# setup.md — Nepali Sign Language (NSL) Recognition Project

Brief for the assistant setting up this project. Read the whole file before running anything. Do the steps in order and stop at every **CHECKPOINT** to confirm the result with the user.

## 0. Scope: setup only (read this first)

This is a **graded individual assignment** (CSC61304, due 24 Nov 2026). The user does the ML work and writes the report. Your only job is to get the machine and project ready.

**You DO:**
- Create the Python environment and install packages (section 2)
- Get PyTorch working on the user's NVIDIA GPU with CUDA (section 2)
- Create the folder structure and a minimal `config.py` (section 3)
- Download the dataset and report basic facts about it (section 4)
- Run the smoke tests (section 5) and report back (section 6)

**You DO NOT:**
- Write, draft, outline, or edit **any part of the report**: no abstract, introduction, literature review, captions, analysis, interpretation, ethics, conclusion, references, or README prose beyond a plain install/run note
- Write or run training, tuning, evaluation, EDA, preprocessing, or demo code
- Generate pseudo-labels, train YOLO, or train any classifier
- Choose models, hyperparameters, or splits for the user
- Fill in `logs/progress_log.md` or any evidence file with content (create empty files only)
- Produce results, metrics, or conclusions of any kind

Section 7 is the user's own plan, included **for reference only**. Do not execute it. If you think a step beyond section 6 would help, **ask the user first**.

**Keep a record of your own work.** Because AI use must be disclosed in the assignment, keep `appendix/ai_usage/setup_commands.md`: a plain list of every command you ran and every file you created, in order, with the output of the CUDA verification. Do not embellish it.

## 1. Project summary

- **Course:** CSC61304 Machine Learning and Intelligent Systems, Individual Assignment (Taylor's University).
- **Goal:** recognise Nepali Sign Language (NSL) characters from a webcam in real time.
- **Planned pipeline (user's design):** webcam frame → YOLO segmentation model separates the hand from the background → hand crop, 224×224 → PyTorch classifier → one of 36 NSL character classes.
- **Framework:** PyTorch only (no TensorFlow/Keras).
- **User's machine:** Windows, PowerShell / Windows Terminal. Use PowerShell syntax. The user says the machine has an NVIDIA GPU that meets CUDA requirements. **Verify this, do not assume it.**

## 2. Environment and CUDA

### 2.1 Python environment

Use **Python 3.11** (3.10–3.12 are acceptable). If the installed Python is newer than what PyTorch wheels support, install 3.11 alongside it.

```powershell
cd <project folder>
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If script execution is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 2.2 Check the GPU and driver

```powershell
nvidia-smi
```

- If this works, note: GPU model, total VRAM, driver version, and the **"CUDA Version"** at the top right. That value is the **highest CUDA version the driver supports**. It is not an installed toolkit.
- If the command is not found or errors, the NVIDIA **driver** is missing or broken. **Stop and tell the user.** Do not try to install or replace GPU drivers or reboot the machine yourself. Point the user to the NVIDIA driver download page or the NVIDIA app and wait for them to confirm.

### 2.3 You do NOT need the full CUDA Toolkit

PyTorch's pip wheels include their own CUDA runtime libraries. For this project you only need:
1. a recent enough **NVIDIA driver** (checked above), and
2. a **CUDA build of PyTorch**.

Do **not** install the standalone CUDA Toolkit or cuDNN installers unless a real, specific error shows they are required. They are not needed to train or run PyTorch/YOLO models here.

### 2.4 Install PyTorch with CUDA (before anything else)

1. Open https://pytorch.org/get-started/locally/ (try to fetch it; if you cannot, use the fallback below) and choose: Stable → Windows → Pip → Python → a **CUDA** compute platform. Pick the CUDA version that is **less than or equal to** the driver's "CUDA Version" from `nvidia-smi`.
2. Run the exact command the selector gives (it uses `--index-url https://download.pytorch.org/whl/cuXXX`). **Do not guess version numbers or index URLs from memory.**
3. `torch` and `torchvision` must come from the same index so they match.

**Fallback if you cannot fetch pytorch.org** (no web access, the page is blocked, or the page content does not include the install command). Work down this list and stop at the first step that works:
1. Try a web search or fetch for https://pytorch.org/get-started/previous-versions/ and read the command from there.
2. **Ask the user to get the command.** Tell them to open https://pytorch.org/get-started/locally/ in their browser, choose Stable → Windows → Pip → Python → the CUDA build that is less than or equal to their driver's "CUDA Version", and paste the command it shows. Then run that command exactly. This is the most reliable route.
3. **Last resort, only with the user's approval:** show the user the exact command you propose, using the pattern `pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX`, where `cuXXX` is a CUDA build tag no higher than the driver supports (for example, `cu126` means CUDA 12.6). Only run it after the user says yes. If pip reports "no matching distribution" or the index does not exist, step down to the next lower CUDA build or go back to step 2. Section 2.6 verification is the safety net: if it fails, the install is wrong.

Never install an unpinned `pip install torch` from the default index on Windows. That gives a CPU-only build. Whichever route you use, record the exact command that worked in `appendix/ai_usage/setup_commands.md`.

### 2.5 Install the other packages

Create `requirements.txt`. Do **not** list `torch` or `torchvision` in it, since those were installed in 2.4.

```
timm
ultralytics
opencv-python
mediapipe
datasets
huggingface_hub
scikit-learn
pandas
numpy
matplotlib
seaborn
optuna
albumentations
tqdm
pyyaml
jupyter
ipykernel
imagehash
pillow
```

```powershell
pip install -r requirements.txt
python -m ipykernel install --user --name nsl --display-name "NSL (venv)"
```

If `mediapipe` fails to install on this Python version, tell the user and continue without it. Do not substitute packages silently.

### 2.6 Verify CUDA (run this after installing everything)

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('cuda available:', torch.cuda.is_available()); print('built with cuda:', torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu'); print('vram GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else 'n/a')"
```

Then a real GPU operation:

```powershell
python -c "import torch; x=torch.randn(2048,2048,device='cuda'); print((x@x).sum().item()); print('ok')"
```

**Troubleshooting if `cuda available` is `False` but `nvidia-smi` works:**
- The version string ends in `+cpu`, or `torch.version.cuda` is `None`: a CPU-only wheel is installed. This usually happens because installing `ultralytics` or another package pulled in its own `torch`. Fix: uninstall torch/torchvision, then reinstall them with the command from 2.4.
- The driver is too old for the chosen CUDA build: choose a lower CUDA build in the selector, or ask the user to update the driver.
- Python version is not supported by the wheel: create the venv with Python 3.11.
- Run `pip list | findstr -i "torch nvidia"` and include the output when reporting to the user.

**CHECKPOINT 1:** Report the `nvidia-smi` summary (GPU, VRAM, driver, max CUDA), the PyTorch/torchvision versions, the CUDA verification output, and any problems. If CUDA cannot be made to work after the troubleshooting above, **stop and report**. Do not carry on silently with the CPU build. (Google Colab is the fallback for training, but only the user decides that.)

## 3. Folder structure

Create these folders (add an empty `.gitkeep` to each) and nothing else:

```
nsl-project/
├─ setup.md
├─ requirements.txt
├─ README.md               # plain install/run note only (venv, install, verify). No analysis or report text
├─ .gitignore
├─ data/
│  ├─ raw/                 # downloaded NSL dataset (do NOT commit)
│  ├─ processed/
│  └─ yolo_hand/
├─ notebooks/
├─ src/
│  └─ config.py            # see below
├─ models/                 # saved weights (do NOT commit large files)
├─ results/
│  ├─ figures/
│  └─ metrics/
├─ logs/
│  └─ progress_log.md      # create EMPTY; the user fills it in
├─ scripts/
│  └─ check_env.py         # see section 5
└─ appendix/
   ├─ screenshots/
   └─ ai_usage/
      └─ setup_commands.md # your record of what you did (section 0)
```

- `.gitignore`: `.venv/`, `data/raw/`, `data/processed/`, `models/*.pt`, `__pycache__/`, `.ipynb_checkpoints/`. Run `git init`.
- `src/config.py` contains **only**: project paths, `SEED = 42`, `IMG_SIZE = 224`, and `CLASS_NAMES` (see section 4). No other code.

## 4. Download the dataset

- **Dataset:** Nepali Sign Language Character Dataset (Birat Poudel et al., 2025), Hugging Face repo ID `Birat-Poudel/Nepali-Sign-Language-Character-Dataset`. Expected URL: https://huggingface.co/datasets/Birat-Poudel/Nepali-Sign-Language-Character-Dataset
- **Paper (user will cite it):** https://arxiv.org/abs/2510.11243
- Download into `data/raw/` with `huggingface_hub.snapshot_download(repo_id=..., repo_type="dataset", local_dir="data/raw")`.
- If the repo ID or link is wrong, **stop and ask the user**. Do not substitute another dataset.
- Expected (verify, do not assume): 36 class folders `0`–`35` in two variants, **Plain Background** and **Random Background**. The dataset page says 1,000 plain and 500 random images per class, while the paper says 1,500 per class in total. Report the actual counts.
- **Class mapping:** the dataset page has a table mapping folder numbers 0–35 to Nepali (Devanagari) characters. Copy it into `config.py` as `CLASS_NAMES` exactly as published. Do not invent or reorder it. If the mapping is missing, tell the user.
- Record the license (the page says MIT) in `README.md` as a single line.

**CHECKPOINT 2:** Report only facts: image count per class and per background variant, image sizes/formats, file types, and any unreadable files. **No analysis or conclusions.**

## 5. Smoke tests

Create `scripts/check_env.py` that prints (and does nothing else):
1. Python, torch, torchvision, cv2, ultralytics, timm versions
2. CUDA availability, GPU name, VRAM
3. Whether one dataset image can be opened with OpenCV and its shape
4. Whether the webcam opens: `cv2.VideoCapture(0).isOpened()` (read one frame, release it, and do not save it)
5. Whether Ultralytics can load a small pretrained segmentation model (for example `yolo11n-seg.pt`; check the Ultralytics docs for current names) and run **one** inference on a single dataset image on the GPU. This only confirms the install. It is not training and not a result.

**CHECKPOINT 3:** Run it and report the output as it is.

## 6. Final report to the user

Reply with a short plain summary of: environment location and Python version; GPU/driver/CUDA result; installed package versions; folder structure created; dataset counts and the class mapping source; smoke-test output; anything unexpected or unresolved. **Then stop and wait.** Do not start section 7.

## 7. Reference plan (the user's own plan. DO NOT EXECUTE)

Kept here only so you understand the design the environment must support.

- **Hand segmentation:** the NSL dataset has class labels only (no boxes or masks), so YOLO-seg cannot train on it directly and COCO models have no "hand" class. The user will decide how to obtain hand labels (auto-generated pseudo-labels or a public hand dataset) and train a single-class `hand` YOLO-seg model. Note that Ultralytics is AGPL-3.0.
- **Classifier input variants:** full image, YOLO crop, masked crop.
- **Splitting:** stratified train/val/test with fixed seed, with near-duplicate checking before the split to avoid leakage.
- **Models to compare:** a CNN transfer-learning baseline (ResNet50 or MobileNetV2) against a transformer (ViT or Swin via `timm`).
- **Tuning:** Optuna on the validation set only.
- **Evaluation:** accuracy, macro-F1, confusion matrix (needs a Devanagari-capable font such as Noto Sans Devanagari in matplotlib), error analysis, and a small self-collected webcam test.
- **Demo:** real-time OpenCV webcam pipeline with temporal smoothing and a confidence threshold.
- **Evidence:** the user keeps their own progress log, screenshots, notebooks (a `.ipynb` link is required in the report appendix), and AI-use screenshots (prompts and results).
