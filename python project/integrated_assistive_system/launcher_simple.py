"""
Simple Launcher for Assistive Navigation System
Clean, simple buttons with guaranteed text visibility
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QGroupBox,
                            QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Import the two main applications
from main import IntegratedAssistiveSystem
from video.video_gui import VideoAnalysisGUI

class SimpleLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Navigation System Launcher")
        self.setGeometry(400, 300, 600, 450)
        self.setFixedSize(600, 450)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize simple launcher interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # Title
        title = QLabel("Assistive Navigation System")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Select Mode:")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle.setFont(subtitle_font)
        main_layout.addWidget(subtitle)
        
        # Button container
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setSpacing(15)
        
        # Live Camera Button
        camera_btn = QPushButton("Live Camera Mode")
        camera_btn.setMinimumHeight(60)
        
        # Simple, clean styling
        camera_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        
        camera_btn.clicked.connect(self.launch_camera)
        button_layout.addWidget(camera_btn)
        
        # Camera description
        camera_desc = QLabel("Real-time camera navigation assistance")
        camera_desc.setAlignment(Qt.AlignCenter)
        camera_desc.setStyleSheet("color: #666; font-size: 12px; margin: 5px;")
        button_layout.addWidget(camera_desc)
        
        # Video Analysis Button
        video_btn = QPushButton("Video Analysis Mode")
        video_btn.setMinimumHeight(60)
        
        video_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        
        video_btn.clicked.connect(self.launch_video)
        button_layout.addWidget(video_btn)
        
        # Video description
        video_desc = QLabel("Analyze video files for path detection")
        video_desc.setAlignment(Qt.AlignCenter)
        video_desc.setStyleSheet("color: #666; font-size: 12px; margin: 5px;")
        button_layout.addWidget(video_desc)
        
        main_layout.addWidget(button_container)
        
        # Add some spacing
        main_layout.addStretch()
        
        # Exit button
        exit_btn = QPushButton("Exit")
        exit_btn.setMaximumWidth(100)
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        
        exit_btn.clicked.connect(self.close)
        
        # Center exit button
        exit_layout = QHBoxLayout()
        exit_layout.addStretch()
        exit_layout.addWidget(exit_btn)
        exit_layout.addStretch()
        
        main_layout.addLayout(exit_layout)
        
    def launch_camera(self):
        """Launch live camera mode"""
        try:
            print("Starting Live Camera Mode...")
            self.hide()
            
            # Create and show camera app
            self.camera_app = IntegratedAssistiveSystem()
            self.camera_app.show()
            
            # Handle camera app closing
            def on_camera_close(event):
                self.show()
                event.accept()
            
            self.camera_app.closeEvent = on_camera_close
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start camera mode: {str(e)}")
            self.show()
    
    def launch_video(self):
        """Launch video analysis mode"""
        try:
            print("Starting Video Analysis Mode...")
            self.hide()
            
            # Run video analysis in separate process
            import subprocess
            import sys
            
            result = subprocess.run([sys.executable, "video_test.py"], 
                                  cwd=os.path.dirname(os.path.abspath(__file__)))
            
            self.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start video mode: {str(e)}")
            self.show()
    
    def closeEvent(self, event):
        """Handle application exit"""
        reply = QMessageBox.question(self, 'Exit', 'Are you sure you want to exit?',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication(sys.argv)
    
    # Create and show simple launcher
    launcher = SimpleLauncher()
    launcher.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
