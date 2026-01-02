#!/usr/bin/env python3
"""
Test script for video analysis
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from video.video_gui import VideoAnalysisGUI

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Create and show video analysis GUI
    video_gui = VideoAnalysisGUI()
    video_gui.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
