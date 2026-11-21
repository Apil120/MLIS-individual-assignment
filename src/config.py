"""Project configuration for NSL Recognition."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "NSL"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
YOLO_HAND_DIR = DATA_DIR / "yolo_hand"

# Model and results paths
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Training constants
SEED = 42
IMG_SIZE = 224

# Class names mapping from folder number to Nepali character
# Source: Birat-Poudel/Nepali-Sign-Language-Character-Dataset (Hugging Face)
CLASS_NAMES = [
    "क",   # 0
    "ख",   # 1
    "ग",   # 2
    "घ",   # 3
    "ङ",   # 4
    "च",   # 5
    "छ",   # 6
    "ज",   # 7
    "झ",   # 8
    "ञ",   # 9
    "ट",   # 10
    "ठ",   # 11
    "ड",   # 12
    "ढ",   # 13
    "ण",   # 14
    "त",   # 15
    "थ",   # 16
    "द",   # 17
    "ध",   # 18
    "न",   # 19
    "प",   # 20
    "फ",   # 21
    "ब",   # 22
    "भ",   # 23
    "म",   # 24
    "य",   # 25
    "र",   # 26
    "ल",   # 27
    "व",   # 28
    "श",   # 29
    "ष",   # 30
    "स",   # 31
    "ह",   # 32
    "क्ष",  # 33
    "त्र",  # 34
    "ज्ञ",  # 35
]

# Verify we have exactly 36 classes
assert len(CLASS_NAMES) == 36, f"Expected 36 classes, got {len(CLASS_NAMES)}"