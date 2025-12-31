"""
Configuration Management for Blind Path Detection System
"""

import os
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).parent

# =========================
# Dataset Configuration
# =========================
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 3
CLASS_NAMES = ["Clear", "Partially Blocked", "Fully Blocked"]
DATA_DIR = r"C:\Users\lenovo\Desktop\PythonCode\FinalProject\dataset"


# =========================
# Model Configuration
# =========================
MODEL_TYPE = "mobilenet"  # "cnn_v1", "cnn_v2", "mobilenet"
USE_TRANSFER = True
OPTIMIZER_TYPE = "adam"  # "adam", "sgd", "rmsprop"
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5
BATCH_NORM_MOMENTUM = 0.9

# =========================
# Training Configuration
# =========================
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 8
LR_PATIENCE = 4
LR_FACTOR = 0.3
USE_CLASS_WEIGHTS = True
USE_DATA_AUGMENTATION = True

# =========================
# Path Configuration
# =========================
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = BASE_DIR / "best_model.h5"
ERROR_SAMPLES_DIR = OUTPUT_DIR / "error_samples"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

# Create necessary directories
for dir_path in [LOG_DIR, OUTPUT_DIR, ERROR_SAMPLES_DIR, GRADCAM_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =========================
# Decision Threshold Configuration
# =========================
THRESHOLDS = {
    "conservative": {
        "full": 0.5,
        "partial": 0.4,
        "clear": 0.75,
        "confidence": 0.5
    },
    "balanced": {
        "full": 0.45,
        "partial": 0.35,
        "clear": 0.7,
        "confidence": 0.45
    },
    "aggressive": {
        "full": 0.35,
        "partial": 0.3,
        "clear": 0.6,
        "confidence": 0.4
    }
}

# =========================
# Audio & Language Configuration
# =========================
LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "ms": "Malay",
    "id": "Indonesian",
    "ar": "Arabic"
}

DEFAULT_LANGUAGE = "en"
DEFAULT_SAFETY_MODE = "conservative"

# =========================
# Navigation Configuration
# =========================
TOLERANCE_ANGLE = 15.0  # degrees
MAX_DEVIATION_TIME = 3.0  # seconds
NAV_UPDATE_INTERVAL = 0.5  # seconds

# =========================
# Performance Configuration
# =========================
INFERENCE_BENCHMARK_RUNS = 100
BENCHMARK_BATCH_SIZES = [1, 4, 8, 16, 32]

# =========================
# Visualization Configuration
# =========================
PLOT_DPI = 150
PLOT_FIGSIZE = (12, 8)
CMAP = "Blues"

# Analysis Configuration
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "analysis"
THRESHOLD_ANALYSIS_DIR = ANALYSIS_OUTPUT_DIR / "threshold"
SPEED_BENCHMARK_DIR = ANALYSIS_OUTPUT_DIR / "benchmark"

# Create analysis directories
for dir_path in [ANALYSIS_OUTPUT_DIR, THRESHOLD_ANALYSIS_DIR, SPEED_BENCHMARK_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)