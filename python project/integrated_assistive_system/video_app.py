"""
Standalone Video Analysis Application
Entry point for video analysis mode
"""

import sys
from PyQt5.QtWidgets import QApplication
from video_gui import VideoAnalysisGUI

def main():
    """Main entry point for video analysis"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show video analysis GUI
    video_gui = VideoAnalysisGUI()
    video_gui.show()
    
    # Run the application
    exit_code = app.exec_()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
