from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFrame, QSlider,
                            QMessageBox, QGridLayout, QGroupBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

class AssistiveGUI(QMainWindow):
    # Signals for communication with main application
    camera_toggled = pyqtSignal(bool)
    detection_toggled = pyqtSignal(bool)
    voice_toggled = pyqtSignal(bool)
    test_voice_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Assistive Navigation System for Blind Users")
        self.setGeometry(100, 100, 900, 700)
        
        # GUI variables
        self.camera_active = False
        self.detection_active = False
        self.voice_active = True
        self.current_frame = None
        self.processed_frame = None
        
        # Status variables
        self.status_text = "System Ready"
        self.instruction_text = "No instruction"
        self.object_count = 0
        
        # Create GUI components
        self.init_ui()
        
        # Bind close event
        self.close_signal = None
        
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
        camera_layout = QVBoxLayout(camera_frame)
        
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: black;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("Camera Feed")
        camera_layout.addWidget(self.camera_label)
        
        # Control panel
        control_frame = QGroupBox("Controls")
        control_layout = QVBoxLayout(control_frame)
        control_frame.setMaximumWidth(250)
        
        # System controls
        controls_label = QLabel("System Controls")
        controls_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        control_layout.addWidget(controls_label)
        
        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.toggle_camera)
        control_layout.addWidget(self.start_camera_btn)
        
        self.start_detection_btn = QPushButton("Start Detection")
        self.start_detection_btn.clicked.connect(self.toggle_detection)
        control_layout.addWidget(self.start_detection_btn)
        
        self.voice_btn = QPushButton("Voice: ON")
        self.voice_btn.clicked.connect(self.toggle_voice)
        control_layout.addWidget(self.voice_btn)
        
        # Status display
        control_layout.addWidget(QLabel())
        status_label = QLabel("Status")
        status_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        control_layout.addWidget(status_label)
        
        self.status_display = QLabel(self.status_text)
        self.status_display.setStyleSheet("color: green;")
        control_layout.addWidget(self.status_display)
        
        # Instruction display
        control_layout.addWidget(QLabel())
        instruction_label = QLabel("Navigation Instruction")
        instruction_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        control_layout.addWidget(instruction_label)
        
        self.instruction_display = QLabel(self.instruction_text)
        self.instruction_display.setStyleSheet("font-weight: bold; font-size: 16px; color: blue;")
        self.instruction_display.setWordWrap(True)
        control_layout.addWidget(self.instruction_display)
        
        # Object count
        control_layout.addWidget(QLabel())
        detection_label = QLabel("Detection Info")
        detection_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        control_layout.addWidget(detection_label)
        
        self.object_count_display = QLabel(f"Objects: {self.object_count}")
        control_layout.addWidget(self.object_count_display)
        
        # Zone indicators
        zone_group = QGroupBox("Zone Status")
        zone_layout = QVBoxLayout(zone_group)
        
        self.zone_labels = {}
        zones = ['Left', 'Center', 'Right']
        for zone in zones:
            label = QLabel(f"{zone}: Clear")
            label.setStyleSheet("color: green;")
            zone_layout.addWidget(label)
            self.zone_labels[zone.lower()] = label
        
        control_layout.addWidget(zone_group)
        
        # Settings
        control_layout.addWidget(QLabel())
        settings_label = QLabel("Settings")
        settings_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        control_layout.addWidget(settings_label)
        
        # Confidence threshold
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setMaximumWidth(150)
        confidence_layout.addWidget(self.confidence_slider)
        control_layout.addLayout(confidence_layout)
        
        # Voice rate
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Voice Speed:"))
        self.voice_rate_slider = QSlider(Qt.Horizontal)
        self.voice_rate_slider.setRange(50, 300)
        self.voice_rate_slider.setValue(150)
        self.voice_rate_slider.setMaximumWidth(150)
        voice_layout.addWidget(self.voice_rate_slider)
        control_layout.addLayout(voice_layout)
        
        # Test button
        control_layout.addWidget(QLabel())
        test_voice_btn = QPushButton("Test Voice")
        test_voice_btn.clicked.connect(self.test_voice)
        control_layout.addWidget(test_voice_btn)
        
        # Add frames to main layout
        main_layout.addWidget(camera_frame, 3)
        main_layout.addWidget(control_frame, 1)
        
    def update_camera_feed(self, frame):
        """Update the camera feed display"""
        if frame is None:
            return
            
        try:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit display
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            
            # Convert to QImage
            height, width, channel = frame_resized.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Convert to QPixmap and display
            pixmap = QPixmap.fromImage(q_image)
            self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
        except Exception as e:
            print(f"Error updating camera feed: {e}")
    
    def update_status(self, status, color='green'):
        """Update status text"""
        self.status_text = status
        self.status_display.setText(status)
        self.status_display.setStyleSheet(f"color: {color};")
    
    def update_instruction(self, instruction):
        """Update navigation instruction"""
        self.instruction_text = instruction
        self.instruction_display.setText(instruction)
    
    def update_object_count(self, count):
        """Update object count display"""
        self.object_count = count
        self.object_count_display.setText(f"Objects: {count}")
    
    def update_zone_status(self, zone_status):
        """Update zone status indicators"""
        for zone, has_objects in zone_status.items():
            if zone in self.zone_labels:
                if has_objects:
                    self.zone_labels[zone].setText(f"{zone.title()}: Object")
                    self.zone_labels[zone].setStyleSheet("color: red;")
                else:
                    self.zone_labels[zone].setText(f"{zone.title()}: Clear")
                    self.zone_labels[zone].setStyleSheet("color: green;")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        self.camera_active = not self.camera_active
        if self.camera_active:
            self.start_camera_btn.setText("Stop Camera")
            self.update_status("Camera Active")
        else:
            self.start_camera_btn.setText("Start Camera")
            self.update_status("Camera Stopped")
            # Clear camera feed
            self.camera_label.clear()
            self.camera_label.setText("Camera Feed")
        
        self.camera_toggled.emit(self.camera_active)
    
    def toggle_detection(self):
        """Toggle detection on/off"""
        self.detection_active = not self.detection_active
        if self.detection_active:
            self.start_detection_btn.setText("Stop Detection")
            self.update_status("Detection Active")
        else:
            self.start_detection_btn.setText("Start Detection")
            self.update_status("Detection Stopped")
        
        self.detection_toggled.emit(self.detection_active)
    
    def toggle_voice(self):
        """Toggle voice guidance on/off"""
        self.voice_active = not self.voice_active
        if self.voice_active:
            self.voice_btn.setText("Voice: ON")
            self.update_status("Voice Enabled")
        else:
            self.voice_btn.setText("Voice: OFF")
            self.update_status("Voice Disabled")
        
        self.voice_toggled.emit(self.voice_active)
    
    def test_voice(self):
        """Test voice functionality"""
        self.update_status("Testing Voice...")
        self.test_voice_requested.emit()
    
    def get_confidence_threshold(self):
        """Get confidence threshold value"""
        return self.confidence_slider.value() / 100.0
    
    def get_voice_rate(self):
        """Get voice rate value"""
        return self.voice_rate_slider.value()
    
    def show_error(self, title, message):
        """Show error message"""
        QMessageBox.critical(self, title, message)
    
    def show_info(self, title, message):
        """Show info message"""
        QMessageBox.information(self, title, message)
    
    def closeEvent(self, event):
        """Handle window closing"""
        reply = QMessageBox.question(self, 'Quit', 
                                   'Do you want to quit the assistive system?',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.camera_active = False
            self.detection_active = False
            self.voice_active = False
            event.accept()
        else:
            event.ignore()
    
    def reset_display(self):
        """Reset display to initial state"""
        self.camera_label.clear()
        self.camera_label.setText("Camera Feed")
        self.update_status("System Ready")
        self.update_instruction("No instruction")
        self.update_object_count(0)
        
        # Reset zone status
        for zone in self.zone_labels:
            self.zone_labels[zone].setText(f"{zone.title()}: Clear")
            self.zone_labels[zone].setStyleSheet("color: green;")
