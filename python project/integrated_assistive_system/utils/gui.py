"""
Integrated GUI System using PyQt5
Combines the best features from both projects
"""

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFrame, QSlider,
                            QMessageBox, QGridLayout, QGroupBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from config import (WINDOW_WIDTH, WINDOW_HEIGHT, CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT,
                   VOICE_MESSAGES, SAFETY_MODES, DEFAULT_SAFETY_MODE)

class AssistiveGUI(QMainWindow):
    # Signals for communication with main application
    camera_toggled = pyqtSignal(bool)
    detection_toggled = pyqtSignal(bool)
    voice_toggled = pyqtSignal(bool)
    safety_mode_changed = pyqtSignal(str)
    test_voice_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Assistive Navigation System")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # GUI variables
        self.camera_active = False
        self.detection_active = False
        self.voice_active = True
        self.current_frame = None
        self.processed_frame = None
        self.current_safety_mode = DEFAULT_SAFETY_MODE
        
        # Status variables
        self.status_text = "System Ready"
        self.instruction_text = "No instruction"
        self.path_status = "Unknown"
        self.confidence_text = "0.00"
        self.fps_text = "0.0"
        
        # Create GUI components
        self.init_ui()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)
        self.update_timer.start(100)  # Update every 100ms
        
    def init_ui(self):
        """Initialize the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Camera feed frame
        camera_frame = QGroupBox("Camera View")
        camera_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_layout = QVBoxLayout(camera_frame)
        
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT)
        self.camera_label.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("Camera Off")
        camera_layout.addWidget(self.camera_label)
        
        # Control panel
        control_frame = QGroupBox("System Controls")
        control_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        control_frame.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_frame)
        
        # System controls
        system_group = QGroupBox("System Controls")
        system_layout = QVBoxLayout(system_group)
        
        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.clicked.connect(self.toggle_camera)
        system_layout.addWidget(self.camera_btn)
        
        self.detection_btn = QPushButton("Start Detection")
        self.detection_btn.clicked.connect(self.toggle_detection)
        system_layout.addWidget(self.detection_btn)
        
        self.voice_btn = QPushButton("Voice: ON")
        self.voice_btn.clicked.connect(self.toggle_voice)
        system_layout.addWidget(self.voice_btn)
        
        control_layout.addWidget(system_group)
        
        # Safety mode controls
        safety_group = QGroupBox("Safety Mode")
        safety_layout = QVBoxLayout(safety_group)
        
        self.safety_mode_btn = QPushButton(f"Mode: {self.current_safety_mode.title()}")
        self.safety_mode_btn.clicked.connect(self.cycle_safety_mode)
        safety_layout.addWidget(self.safety_mode_btn)
        
        control_layout.addWidget(safety_group)
        
        # Status display
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        self.path_status_label = QLabel("Path: Unknown")
        self.path_status_label.setStyleSheet("color: blue; font-weight: bold;")
        status_layout.addWidget(self.path_status_label)
        
        self.confidence_label = QLabel("Confidence: 0.00")
        status_layout.addWidget(self.confidence_label)
        
        self.fps_label = QLabel("FPS: 0.0")
        status_layout.addWidget(self.fps_label)
        
        control_layout.addWidget(status_group)
        
        # Navigation instruction
        instruction_group = QGroupBox("Navigation Instruction")
        instruction_layout = QVBoxLayout(instruction_group)
        
        self.instruction_label = QLabel("No instruction")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("font-size: 14px; font-weight: bold; color: blue; padding: 10px;")
        self.instruction_label.setMinimumHeight(60)
        instruction_layout.addWidget(self.instruction_label)
        
        control_layout.addWidget(instruction_group)
        
        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Voice rate slider
        voice_rate_layout = QHBoxLayout()
        voice_rate_layout.addWidget(QLabel("Voice Speed:"))
        self.voice_rate_slider = QSlider(Qt.Horizontal)
        self.voice_rate_slider.setRange(50, 300)
        self.voice_rate_slider.setValue(150)
        self.voice_rate_slider.valueChanged.connect(self.on_voice_rate_changed)
        voice_rate_layout.addWidget(self.voice_rate_slider)
        self.voice_rate_value = QLabel("150")
        voice_rate_layout.addWidget(self.voice_rate_value)
        settings_layout.addLayout(voice_rate_layout)
        
        # Test voice button
        test_voice_btn = QPushButton("Test Voice")
        test_voice_btn.clicked.connect(self.test_voice)
        settings_layout.addWidget(test_voice_btn)
        
        control_layout.addWidget(settings_group)
        
        # Add frames to main layout
        main_layout.addWidget(camera_frame, 3)
        main_layout.addWidget(control_frame, 1)
        
    def toggle_camera(self):
        """Toggle camera on/off"""
        self.camera_active = not self.camera_active
        if self.camera_active:
            self.camera_btn.setText("Stop Camera")
            self.status_label.setText("Status: Camera Active")
            self.camera_label.setText("")
        else:
            self.camera_btn.setText("Start Camera")
            self.status_label.setText("Status: Camera Stopped")
            self.camera_label.setText("Camera Off")
            self.camera_label.setPixmap(QPixmap())
        
        self.camera_toggled.emit(self.camera_active)
        
    def toggle_detection(self):
        """Toggle detection on/off"""
        self.detection_active = not self.detection_active
        if self.detection_active:
            self.detection_btn.setText("Stop Detection")
            self.status_label.setText("Status: Detection Active")
        else:
            self.detection_btn.setText("Start Detection")
            self.status_label.setText("Status: Detection Stopped")
        
        self.detection_toggled.emit(self.detection_active)
        
    def toggle_voice(self):
        """Toggle voice guidance on/off"""
        self.voice_active = not self.voice_active
        if self.voice_active:
            self.voice_btn.setText("Voice: ON")
            self.status_label.setText("Status: Voice Enabled")
        else:
            self.voice_btn.setText("Voice: OFF")
            self.status_label.setText("Status: Voice Disabled")
        
        self.voice_toggled.emit(self.voice_active)
        
    def cycle_safety_mode(self):
        """Cycle through safety modes"""
        modes = list(SAFETY_MODES.keys())
        current_index = modes.index(self.current_safety_mode)
        next_index = (current_index + 1) % len(modes)
        self.current_safety_mode = modes[next_index]
        
        self.safety_mode_btn.setText(f"Mode: {self.current_safety_mode.title()}")
        self.safety_mode_changed.emit(self.current_safety_mode)
        
    def on_voice_rate_changed(self, value):
        """Handle voice rate slider change"""
        self.voice_rate_value.setText(str(value))
        # This will be handled by the main application
        
    def test_voice(self):
        """Test voice functionality"""
        self.test_voice_requested.emit()
        
    def update_camera_feed(self, frame):
        """Update the camera feed display"""
        if frame is None:
            return
        
        self.current_frame = frame.copy()
        
    def update_displays(self):
        """Update all displays (called by timer)"""
        # Update camera feed
        if self.current_frame is not None:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit display
            h, w = frame_rgb.shape[:2]
            if w > CAMERA_DISPLAY_WIDTH or h > CAMERA_DISPLAY_HEIGHT:
                scale = min(CAMERA_DISPLAY_WIDTH / w, CAMERA_DISPLAY_HEIGHT / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            
            # Convert to QImage
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Convert to QPixmap and display
            pixmap = QPixmap.fromImage(q_image)
            self.camera_label.setPixmap(pixmap)
            
    def update_status(self, status, color='green'):
        """Update status text"""
        self.status_label.setText(f"Status: {status}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
    def update_path_status(self, path_status, confidence):
        """Update path status display"""
        self.path_status = path_status
        self.confidence_text = f"{confidence:.2f}"
        
        # Set color based on status
        colors = {
            "Clear": "green",
            "Partially Blocked": "orange", 
            "Fully Blocked": "red"
        }
        color = colors.get(path_status, "gray")
        
        self.path_status_label.setText(f"Path: {path_status}")
        self.path_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.confidence_label.setText(f"Confidence: {self.confidence_text}")
        
    def update_instruction(self, instruction):
        """Update navigation instruction"""
        self.instruction_text = instruction
        self.instruction_label.setText(instruction)
        
    def update_fps(self, fps):
        """Update FPS display"""
        self.fps_text = f"{fps:.1f}"
        self.fps_label.setText(f"FPS: {self.fps_text}")
        
    def get_voice_rate(self):
        """Get voice rate value"""
        return self.voice_rate_slider.value()
        
    def get_safety_mode(self):
        """Get current safety mode"""
        return self.current_safety_mode
        
    def show_error(self, title, message):
        """Show error message"""
        QMessageBox.critical(self, title, message)
        
    def show_info(self, title, message):
        """Show info message"""
        QMessageBox.information(self, title, message)
        
    def closeEvent(self, event):
        """Handle window closing"""
        reply = QMessageBox.question(self, 'Quit', 'Do you want to quit the assistive system?',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.camera_active = False
            self.detection_active = False
            self.voice_active = False
            event.accept()
        else:
            event.ignore()
