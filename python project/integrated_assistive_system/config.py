"""
Integrated Assistive Navigation System Configuration
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best_model.h5"

# Camera settings
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Model settings (MobileNet from academic project)
MODEL_INPUT_SHAPE = (224, 224, 3)
MODEL_CLASSES = ["Clear", "Partially Blocked", "Fully Blocked"]
CONFIDENCE_THRESHOLD = 0.5

# Safety modes (from academic project)
SAFETY_MODES = {
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

DEFAULT_SAFETY_MODE = "conservative"

# Voice settings
VOICE_ENABLED = True
VOICE_RATE = 150  # Words per minute
VOICE_VOLUME = 0.8

# GUI settings
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
CAMERA_DISPLAY_WIDTH = 640
CAMERA_DISPLAY_HEIGHT = 480

# System settings
PROCESSING_FPS = 30
INSTRUCTION_DELAY = 3.0  # Minimum seconds between voice instructions

# Colors for different safety levels
SAFETY_COLORS = {
    "Clear": (0, 255, 0),      # Green
    "Partially Blocked": (0, 255, 255),  # Yellow
    "Fully Blocked": (0, 0, 255)  # Red
}

# Voice messages
VOICE_MESSAGES = {
    "Clear": "Path clear",
    "Partially Blocked": "Warning obstacle ahead",
    "Fully Blocked": "Stop way blocked",
    "Go straight": "Go straight",
    "Go left": "Go left",
    "Go right": "Go right",
    "Stop": "Stop",
    "Proceed with caution": "Proceed with caution",
    "Stop and find another way": "Stop and find another way",
    "Stop and turn around": "Stop and turn around"
}
