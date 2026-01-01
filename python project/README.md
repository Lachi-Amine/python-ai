# Integrated Assistive Navigation System

A comprehensive assistive navigation system for visually impaired individuals that combines advanced path detection with multiple analysis modes.

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Camera (built-in or USB)
- Virtual environment (recommended)

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd "integrated_assistive_system"
   ```

2. **Activate virtual environment:**
   ```bash
   source ../.venv/bin/activate  # On Windows: ..\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the launcher:**
   ```bash
   python launcher.py
   ```

## Project Structure

```
integrated_assistive_system/
├── launcher.py          # Main launcher and mode selection
├── main.py              # Live camera application controller
├── config.py            # Configuration settings
├── core/                # Core detection and navigation modules
│   ├── path_detector.py # MobileNet-based path classification
│   ├── zone_detector.py # Zone-based obstacle detection
│   └── navigation.py    # Directional navigation logic
├── utils/               # Utility modules
│   ├── camera.py        # Camera capture and management
│   ├── voice_guide.py   # Text-to-speech navigation instructions
│   └── gui.py           # Live camera interface
├── video/               # Video analysis modules
│   ├── video_analyzer.py # Video file analysis engine
│   ├── video_gui.py     # Video analysis interface
│   └── video_app.py     # Video application entry point
├── models/              # Model files
│   └── best_model.h5    # Trained MobileNet model (linked from ../final model/)
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Features

- **🧠 Smart Path Detection**: MobileNetV2 model with 3-level safety classification
- **🎯 Dual Analysis Modes**: Live camera and video analysis
- **🎮 Intelligent Navigation**: Directional guidance with voice instructions
- **🖥️ Modern Interface**: PyQt5 GUI with real-time feedback
- **📊 Video Analysis**: Batch processing with export options

## Usage

### Launcher Menu
Choose between:
1. **📷 Live Camera Mode** - Real-time navigation assistance
2. **🎬 Video Analysis Mode** - Analyze video files

### Live Camera Mode
1. Click **"Start Camera"** to activate the camera
2. Click **"Start Detection"** to begin path analysis  
3. Listen for **directional instructions**: "Go left", "Go right", "Go straight"
4. Adjust **Safety Mode** (Conservative/Balanced/Aggressive)
5. Toggle **Voice** on/off as needed

### Video Analysis Mode
1. Click **"Browse Video"** to select a video file
2. Click **"Analyze Video"** to start processing
3. Watch **real-time progress** and frame-by-frame results
4. Use **Results tab** to navigate through analyzed frames
5. **Export** results to CSV or create annotated video

---

**Version**: 2.0 Organized  
**Last Updated**: 2025  
**Compatibility**: Python 3.8+, macOS, Windows, Linux
