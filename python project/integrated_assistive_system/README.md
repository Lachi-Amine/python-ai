# Integrated Assistive Navigation System

A comprehensive assistive navigation system for visually impaired individuals that combines advanced path detection with multiple analysis modes.

## Features

### 🧠 **Smart Path Detection**
- **MobileNetV2 Model**: Uses a trained deep learning model for accurate path classification
- **3-Level Safety Classification**: Clear, Partially Blocked, Fully Blocked
- **Adaptive Safety Modes**: Conservative, Balanced, and Aggressive detection sensitivity
- **Real-time Processing**: Live camera feed with instant path status updates

### 🎯 **Dual Analysis Modes**
- **📷 Live Camera Mode**: Real-time navigation with instant feedback
- **🎬 Video Analysis Mode**: Process video files with detailed analysis and export

### 🎮 **Intelligent Navigation**
- **Directional Guidance**: "Go left", "Go right", "Go straight" instructions
- **Voice Guidance**: Clear audio instructions for path status
- **Visual Feedback**: Color-coded status indicators with zone detection
- **Confidence Scoring**: Probability-based decision making
- **Emergency Detection**: Priority alerts for immediate dangers

### 🖥️ **Modern Interface**
- **PyQt5 GUI**: Clean, responsive user interface
- **Launcher Menu**: Easy mode selection
- **Live Camera Feed**: Real-time video display with status overlays
- **Video Preview**: Frame-by-frame analysis with navigation
- **System Controls**: Easy toggles for camera, detection, and voice
- **Performance Monitoring**: FPS counter and confidence indicators

### 📊 **Video Analysis Features**
- **Batch Processing**: Analyze entire video files
- **Progress Tracking**: Real-time analysis progress
- **Frame Navigation**: Jump to any frame for detailed inspection
- **Export Options**: CSV results and annotated video output
- **Statistics**: Comprehensive analysis summaries

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

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the launcher:**
   ```bash
   python launcher.py
   ```

## 🎮 **How to Use**

### **Launcher Menu**
When you run the system, you'll see a launcher with two options:

1. **📷 Live Camera Mode** - Real-time navigation assistance
2. **🎬 Video Analysis Mode** - Analyze video files

### **Live Camera Mode**
1. Click **"Start Camera"** to activate the camera
2. Click **"Start Detection"** to begin path analysis  
3. Listen for **directional instructions**: "Go left", "Go right", "Go straight"
4. Adjust **Safety Mode** (Conservative/Balanced/Aggressive)
5. Toggle **Voice** on/off as needed

### **Video Analysis Mode**
1. Click **"Browse Video"** to select a video file
2. Click **"Analyze Video"** to start processing
3. Watch **real-time progress** and frame-by-frame results
4. Use **Results tab** to navigate through analyzed frames
5. **Export** results to CSV or create annotated video

## 🎯 **Directional Navigation**

The system now provides **specific directional guidance**:

- **🟢 Clear Path** → "Go straight"
- **🟡 Left Obstacle** → "Go right"  
- **🟡 Right Obstacle** → "Go left"
- **🔴 Center Block** → "Stop" or "Find another way"

## 📊 **Video Analysis Capabilities**

### **Supported Formats**
- MP4, AVI, MOV, MKV, WMV
- Any resolution with automatic scaling

### **Analysis Features**
- **Frame-by-frame path detection**
- **Directional instruction for each frame**
- **Zone-based obstacle detection**
- **Confidence scoring and probabilities**

### **Export Options**
- **CSV Export**: Frame-by-frame results with timestamps
- **Video Export**: Annotated video with status overlays
- **Statistics Summary**: Path status distribution and instruction frequency

## System Architecture

### Core Components
- **`launcher.py`**: Main launcher and mode selection
- **`main.py`**: Live camera application controller
- **`video_analyzer.py`**: Video file analysis engine
- **`video_gui.py`**: Video analysis interface
- **`path_detector.py`**: MobileNet-based path classification
- **`zone_detector.py`**: Zone-based obstacle detection
- **`navigation.py`**: Directional navigation logic
- **`camera.py`**: Camera capture and management
- **`voice_guide.py`**: Text-to-speech navigation instructions
- **`gui.py`**: Live camera interface
- **`config.py`**: System configuration and settings

### Model Information
- **Model**: MobileNetV2 with transfer learning
- **Input**: 224x224x3 RGB images
- **Output**: 3-class path classification
- **Training**: Optimized for indoor/outdoor navigation scenarios

## Performance

### System Requirements
- **CPU**: Standard modern processor (Intel i5 or equivalent)
- **RAM**: 4GB minimum, 8GB recommended
- **Camera**: 640x480 resolution at 30 FPS
- **Storage**: 500MB for application and model

### Benchmarks
- **Inference Time**: ~50ms per frame
- **Processing FPS**: 20-30 FPS (depending on hardware)
- **Accuracy**: 85-90% on test scenarios
- **Response Time**: <200ms from capture to voice output

## Troubleshooting

### Common Issues

**Camera not found:**
- Check camera connection
- Ensure camera is not used by another application
- Try different camera index (change `CAMERA_ID` in config.py)

**Model loading error:**
- Verify `models/best_model.h5` exists
- Check TensorFlow installation
- Ensure sufficient system memory

**Voice not working:**
- Check system audio output
- Verify pyttsx3 installation
- Try different voice in system settings

**Performance issues:**
- Reduce camera resolution in config.py
- Close other applications
- Use "Aggressive" safety mode for faster processing

### Advanced Configuration

Edit `config.py` to customize:
- Camera settings (resolution, FPS)
- Model thresholds and sensitivity
- Voice parameters (rate, volume)
- GUI dimensions and colors

## Development

### Project Structure
```
integrated_assistive_system/
├── main.py              # Main application
├── config.py            # Configuration settings
├── path_detector.py     # Path detection logic
├── camera.py            # Camera interface
├── voice_guide.py       # Voice guidance
├── gui.py               # User interface
├── models/              # Model files
│   └── best_model.h5    # Trained MobileNet model
├── requirements.txt     # Dependencies
└── README.md           # This file
```

### Contributing
1. Test changes with virtual environment
2. Maintain backward compatibility
3. Update documentation for new features
4. Follow PEP 8 coding standards

## License

This project combines components from academic research and open-source development. See individual component licenses for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review system requirements
3. Test with different camera setups
4. Verify model file integrity

---

**Version**: 1.0 Integrated  
**Last Updated**: 2025  
**Compatibility**: Python 3.8+, macOS, Windows, Linux
