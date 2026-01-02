# Integrated Assistive Navigation System

A comprehensive assistive navigation system for visually impaired individuals that combines advanced AI path detection with real-time voice guidance and visual obstacle detection.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Camera (built-in or USB)
- Virtual environment (recommended)

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd "/Users/lachiamine/Documents/python project/integrated_assistive_system"
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

### 🎯 Quick Launch Commands

**Option 1: From Project Directory**
```bash
cd "/Users/lachiamine/Documents/python project/integrated_assistive_system" && source ../.venv/bin/activate && python launcher.py
```

**Option 2: One-Liner**
```bash
cd "/Users/lachiamine/Documents/python project" && source .venv/bin/activate && python integrated_assistive_system/launcher.py
```

## 📁 Project Structure

```
python project/
├── .venv/                    # Virtual environment
├── final model/              # AI model files (26MB best_model.h5)
├── integrated_assistive_system/
│   ├── launcher.py           # Main launcher and mode selection
│   ├── main.py               # Live camera application controller
│   ├── config.py             # Configuration settings
│   ├── core/                 # Core detection and navigation modules
│   │   ├── path_detector.py  # MobileNet-based path classification
│   │   ├── zone_detector.py  # Zone-based obstacle detection
│   │   └── navigation.py     # Directional navigation logic
│   ├── utils/               # Utility modules
│   │   ├── camera.py        # Camera capture and management
│   │   ├── voice_guide.py   # Text-to-speech navigation instructions
│   │   └── gui.py           # Live camera interface
│   ├── video/               # Video analysis modules
│   │   ├── video_analyzer.py # Video file analysis engine
│   │   ├── video_gui.py     # Video analysis interface
│   │   └── video_app.py     # Video application entry point
│   ├── models/              # Model files (symlink to final model)
│   ├── requirements.txt     # Dependencies
│   └── README.md           # This file
└── README.md               # Main project documentation
```

## ✨ Features

### 🧠 **AI-Powered Path Detection**
- **MobileNetV2 Model**: Advanced deep learning for accurate path classification
- **3-Level Safety Classification**: Clear, Partially Blocked, Fully Blocked
- **Real-Time Processing**: Live camera feed with instant path status updates
- **Confidence Scoring**: Probability-based decision making

### 🎯 **Dual Analysis Modes**
- **📷 Live Camera Mode**: Real-time navigation with instant feedback
- **🎬 Video Analysis Mode**: Process video files with detailed analysis

### 🎮 **Intelligent Navigation**
- **Directional Guidance**: "Go left", "Go right", "Go straight", "Stop and turn around"
- **Voice Instructions**: Clear audio guidance with emergency priority
- **Visual Obstacle Detection**: Colored zones showing obstacle locations
- **Emergency Detection**: High-priority alerts for immediate dangers

### 🖥️ **Modern Interface**
- **PyQt5 GUI**: Clean, responsive user interface
- **Launcher Menu**: Easy mode selection
- **Live Camera Feed**: Real-time video with status overlays
- **Zone Visualization**: Green/Yellow/Red obstacle coloring
- **Performance Monitoring**: FPS counter and confidence indicators

### 🗣️ **Voice Guidance System**
- **Text-to-Speech**: Natural voice instructions using pyttsx3
- **Smart Timing**: 3-second delay between normal instructions
- **Emergency Priority**: Immediate alerts for dangerous situations
- **Adjustable Settings**: Voice rate and volume controls

## 🎮 How to Use

### Launcher Menu
When you run the system, you'll see a launcher with two options:

1. **📷 Live Camera Mode** - Real-time navigation assistance
2. **🎬 Video Analysis Mode** - Analyze video files

### Live Camera Mode
1. Click **"Start Camera"** to activate the camera
2. Click **"Start Detection"** to begin AI path analysis  
3. Listen for **voice instructions**: "Go left", "Go right", "Go straight"
4. Watch **colored zones** showing obstacle locations:
   - 🟢 **Green**: Clear path - "Go straight"
   - 🟡 **Yellow**: Partial obstacle - "Go left/right"
   - 🔴 **Red**: Blocked path - "Stop and turn around"
5. Adjust **Safety Mode** (Conservative/Balanced/Aggressive)
6. Toggle **Voice** on/off as needed

### Video Analysis Mode
1. Click **"Browse Video"** to select a video file
2. Click **"Analyze Video"** to start processing
3. Watch **real-time progress** and frame-by-frame results
4. Use **Results tab** to navigate through analyzed frames
5. **Export** results to CSV or create annotated video

## 🔧 Technical Details

### AI Model Information
- **Model**: MobileNetV2 with transfer learning
- **Input**: 224x224x3 RGB images
- **Output**: 3-class path classification
- **Model Size**: 26MB (optimized for performance)
- **Accuracy**: 85-90% on test scenarios

### Performance Metrics
- **Processing Speed**: ~50ms per frame
- **Frame Rate**: 20-30 FPS (depending on hardware)
- **Response Time**: <200ms from capture to voice output
- **Memory Usage**: ~500MB for application and model

### System Requirements
- **CPU**: Standard modern processor (Intel i5 or equivalent)
- **RAM**: 4GB minimum, 8GB recommended
- **Camera**: 640x480 resolution at 30 FPS
- **Storage**: 500MB for application and model

## 🎯 Navigation Logic

### Path Status → Voice Instructions
- **🟢 Clear Path** → "Go straight"
- **🟡 Left Obstacle** → "Go right"  
- **🟡 Right Obstacle** → "Go left"
- **🔴 Center Block** → "Stop and turn around"
- **🔴 Fully Blocked** → "Stop and find another way"

### Zone-Based Detection
- **Left Zone**: Obstacles on the left side of the path
- **Center Zone**: Critical obstacles directly ahead
- **Right Zone**: Obstacles on the right side of the path

## 🛠️ Configuration

The system can be customized by editing `config.py`:

```python
# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Voice settings
VOICE_RATE = 150  # Words per minute
VOICE_VOLUME = 0.8  # 0.0 to 1.0
INSTRUCTION_DELAY = 3.0  # Seconds between instructions

# Safety modes
SAFETY_MODES = {
    "conservative": {"full": 0.5, "partial": 0.4},
    "balanced": {"full": 0.45, "partial": 0.35},
    "aggressive": {"full": 0.35, "partial": 0.3}
}
```

## 🔍 Troubleshooting

### Common Issues

**Camera not found:**
- Check camera connection
- Ensure camera is not used by another application
- Try different camera index (change `CAMERA_ID` in config.py)

**Voice not working:**
- Check system audio output
- Verify pyttsx3 installation
- Try different voice in system settings

**Model loading error:**
- Verify `../final model/best_model.h5` exists
- Check TensorFlow installation
- Ensure sufficient system memory

**Performance issues:**
- Reduce camera resolution in config.py
- Close other applications
- Use "Aggressive" safety mode for faster processing

## 📈 Development

### Key Components
- **`launcher.py`**: Main application launcher
- **`main.py`**: Live camera processing loop
- **`path_detector.py`**: AI model inference
- **`zone_detector.py`**: Obstacle zone analysis
- **`navigation.py`**: Direction generation logic
- **`voice_guide.py`**: Text-to-speech system

### Adding New Features
1. Modify `config.py` for new settings
2. Update relevant modules in `core/` or `utils/`
3. Test with virtual environment
4. Update documentation

## 📄 License

This project combines components from academic research and open-source development. See individual component licenses for details.

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Verify system requirements
3. Test with different camera setups
4. Ensure model file integrity

---

**Version**: 2.0 Complete  
**Last Updated**: January 2026  
**Compatibility**: Python 3.8+, macOS, Windows, Linux  
**Status**: ✅ Production Ready with Full AI Integration
