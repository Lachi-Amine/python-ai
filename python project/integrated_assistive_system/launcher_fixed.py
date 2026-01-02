"""
Integrated Assistive System Launcher - Fixed Version
Choose between live camera mode and video analysis mode
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFrame, QGroupBox,
                            QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QIcon

# Import the two main applications
from main import IntegratedAssistiveSystem
from video.video_gui import VideoAnalysisGUI

class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Assistive Navigation System - Launcher")
        self.setGeometry(300, 200, 500, 400)
        self.setFixedSize(500, 400)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the launcher interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        
        # Title
        title_label = QLabel("Assistive Navigation System")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 18, QFont.Bold)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Choose your analysis mode:")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_font = QFont("Arial", 12)
        subtitle_label.setFont(subtitle_font)
        main_layout.addWidget(subtitle_label)
        
        # Mode selection buttons
        modes_group = QGroupBox("Select Mode")
        modes_layout = QVBoxLayout(modes_group)
        modes_layout.setSpacing(15)
        
        # Live Camera Mode
        camera_btn = QPushButton("LIVE CAMERA MODE")
        camera_btn.setMinimumHeight(80)
        camera_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                font-family: Arial, sans-serif;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        camera_btn.clicked.connect(self.launch_camera_mode)
        modes_layout.addWidget(camera_btn)
        
        camera_desc = QLabel("Real-time camera analysis with live navigation guidance")
        camera_desc.setAlignment(Qt.AlignCenter)
        camera_desc.setStyleSheet("color: #666; font-size: 11px; margin-bottom: 10px;")
        modes_layout.addWidget(camera_desc)
        
        # Video Analysis Mode
        video_btn = QPushButton("VIDEO ANALYSIS MODE")
        video_btn.setMinimumHeight(80)
        video_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                font-family: Arial, sans-serif;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        video_btn.clicked.connect(self.launch_video_mode)
        modes_layout.addWidget(video_btn)
        
        video_desc = QLabel("Analyze video files with detailed path detection and export options")
        video_desc.setAlignment(Qt.AlignCenter)
        video_desc.setStyleSheet("color: #666; font-size: 11px;")
        modes_layout.addWidget(video_desc)
        
        main_layout.addWidget(modes_group)
        
        # Info section
        info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel("""Features:
• MobileNetV2 path detection
• Directional navigation (Go Left/Right/Straight)
• Voice guidance with safety modes
• Real-time performance monitoring
• Video export with analysis overlays
• CSV export of results""")
        info_text.setStyleSheet("font-size: 10px; color: #333;")
        info_layout.addWidget(info_text)
        
        main_layout.addWidget(info_group)
        
        # Exit button
        exit_btn = QPushButton("Exit")
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        exit_btn.clicked.connect(self.close)
        main_layout.addWidget(exit_btn)
        
    def launch_camera_mode(self):
        """Launch live camera mode"""
        try:
            print("Launching Live Camera Mode...")
            
            # Hide launcher
            self.hide()
            
            # Create and run camera application
            self.camera_app = IntegratedAssistiveSystem()
            self.camera_app.show()
            
            # Show launcher again when camera app closes
            self.camera_app.closeEvent = lambda event: self.show()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch camera mode: {str(e)}")
            self.show()
    
    def launch_video_mode(self):
        """Launch video analysis mode"""
        try:
            print("Launching Video Analysis Mode...")
            
            # Hide launcher
            self.hide()
            
            # Create and run video analysis application in a separate process
            import subprocess
            import sys
            
            # Run video analysis as separate process to avoid Qt conflicts
            result = subprocess.run([sys.executable, "video_test.py"], 
                                  cwd=os.path.dirname(os.path.abspath(__file__)))
            
            # Show launcher again when video app closes
            self.show()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch video mode: {str(e)}")
            self.show()
    
    def closeEvent(self, event):
        """Handle window closing"""
        reply = QMessageBox.question(self, 'Exit', 'Are you sure you want to exit?',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    launcher = LauncherWindow()
    launcher.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
