# Nepali Sign Language (NSL) Recognition Project

## Setup Instructions

### 1. Create and activate virtual environment
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch with CUDA (for RTX 4060, CUDA 12.6)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install other dependencies
```powershell
pip install -r requirements.txt
python -m ipykernel install --user --name nsl --display-name "NSL (venv)"
```

### 4. Verify CUDA
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 5. Download dataset
```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Birat-Poudel/Nepali-Sign-Language-Character-Dataset', repo_type='dataset', local_dir='data/raw')"
```

### 6. Extract dataset
```powershell
python -c "import zipfile; z = zipfile.ZipFile('data/raw/NSL.zip'); z.extractall('data/raw'); z.close()"
```

## Dataset
- **Source**: Birat-Poudel/Nepali-Sign-Language-Character-Dataset (Hugging Face)
- **License**: MIT
- **Classes**: 36 Nepali characters (0-35)
- **Variants**: Plain Background (1000 images/class), Random Background (500 images/class)
- **Total**: 54,000 images (36,000 plain + 18,000 random)
- **Image size**: 300x300 JPEG RGB

## Class Mapping
| Folder | Character | Folder | Character |
|--------|-----------|--------|-----------|
| 0 | क | 18 | ध |
| 1 | ख | 19 | न |
| 2 | ग | 20 | प |
| 3 | घ | 21 | फ |
| 4 | ङ | 22 | ब |
| 5 | च | 23 | भ |
| 6 | छ | 24 | म |
| 7 | ज | 25 | य |
| 8 | झ | 26 | र |
| 9 | ञ | 27 | ल |
| 10 | ट | 28 | व |
| 11 | ठ | 29 | श |
| 12 | ड | 30 | ष |
| 13 | ढ | 31 | स |
| 14 | ण | 32 | ह |
| 15 | त | 33 | क्ष |
| 16 | थ | 34 | त्र |
| 17 | द | 35 | ज्ञ |

## Run Smoke Test
```powershell
python scripts/check_env.py
```