# Blind Path Obstacle Detection System

##  Project Overview
**Course:** AIT102 - Python and TensorFlow Programming  
**Group:** 6  

This project is a real-time obstacle detection system designed to assist visually impaired individuals. It leverages a pre-trained **MobileNetV2** deep learning model to classify environments into safety levels and provides immediate audio-visual feedback.

The system is divided into two functional modules:
1.  **Core Detection System (Backend):** Handles real-time computer vision inference and safety logic.
2.  **User Interface Prototype (Frontend):** Demonstrates the accessible design concept for the end-user application.

---

## Key Features

### 1. Real-time Obstacle Detection
* **High Performance:** Uses MobileNetV2 for fast inference on standard CPU/GPU.
* **Live FPS Monitoring:** Real-time feedback on system latency.

### 2. Intelligent Decision Logic (3-Level Safety)
The system classifies the current path into three distinct states:
* 🟢 **Green (Safe):** Path is clear. No disturbance.
* 🟡 **Yellow (Warning):** Partial obstacle detected (e.g., side objects). Gentle audio warning.
* 🔴 **Red (Danger):** Full blockage detected (e.g., walls, close objects). Urgent "STOP" alert.

### 3. Dynamic Safety Modes
Users can switch detection sensitivity based on their environment:
* **Conservative Mode:** High sensitivity (best for unfamiliar/crowded areas).
* **Balanced Mode:** Standard sensitivity (best for daily use).
* **Aggressive Mode:** Low sensitivity (reduces false alarms in familiar areas).

### 4. Audio Navigation
* Offline Text-to-Speech (TTS) integration using `pyttsx3`.
* Provides clear voice commands ("Stop", "Warning", "Path Clear").

---

## Installation Requirements

### Prerequisites
* Python 3.8 or higher
* Webcam (for real-time input)

### Dependencies
Install the necessary libraries using pip:

```bash
pip install tensorflow opencv-python numpy pyttsx3