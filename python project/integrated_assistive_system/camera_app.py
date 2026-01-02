#!/usr/bin/env python3
"""
Camera Application - Simplified version for launcher
Handles proper exit and return to launcher
"""

import sys
import os
import signal
import threading

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from utils.gui import AssistiveGUI
from utils.camera import CameraCapture
from core.path_detector import PathDetector
from core.zone_detector import ZoneDetector
from core.navigation import NavigationAssistant
from utils.voice_guide import VoiceGuide
from config import (VOICE_MESSAGES, SAFETY_COLORS, PROCESSING_FPS, INSTRUCTION_DELAY,
                   VOICE_ENABLED, VOICE_RATE, VOICE_VOLUME)

class CameraApp:
    def __init__(self):
        print("Initializing Camera Application...")
        
        # Create PyQt application
        self.app = QApplication(sys.argv)
        self.gui = AssistiveGUI()
        
        # Initialize system components
        self.camera = CameraCapture()
        self.path_detector = PathDetector()
        self.zone_detector = ZoneDetector()
        self.navigation = NavigationAssistant()
        self.voice_guide = VoiceGuide() if VOICE_ENABLED else None
        
        # Connect GUI signals
        self.gui.camera_toggled.connect(self.on_camera_toggled)
        self.gui.detection_toggled.connect(self.on_detection_toggled)
        self.gui.voice_toggled.connect(self.on_voice_toggled)
        
        # Processing control
        self.is_running = False
        self.processing_thread = None
        
        # Setup signal handlers for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("Camera Application initialized")
    
    def signal_handler(self, signum, frame):
        """Handle system signals for clean exit"""
        print(f"Received signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)
    
    def on_camera_toggled(self, active):
        """Handle camera toggle"""
        if active:
            if self.camera.initialize():
                self.gui.update_status("Camera Active", "green")
                print("Camera activated")
            else:
                self.gui.update_status("Camera Error", "red")
                self.gui.set_camera_active(False)
        else:
            self.camera.release()
            self.gui.update_status("Camera Inactive", "orange")
            print("Camera deactivated")
    
    def on_detection_toggled(self, active):
        """Handle detection toggle"""
        if active and self.camera.is_camera_available():
            self.is_running = True
            self.processing_thread = threading.Thread(target=self.main_loop, daemon=True)
            self.processing_thread.start()
            self.gui.update_status("Detection Active", "green")
            print("Detection activated")
        else:
            self.is_running = False
            self.gui.update_status("Detection Inactive", "orange")
            print("Detection deactivated")
    
    def on_voice_toggled(self, active):
        """Handle voice toggle"""
        self.gui.voice_active = active
        status = "Voice Active" if active else "Voice Inactive"
        color = "green" if active else "orange"
        self.gui.update_status(status, color)
        print(f"Voice {status.lower()}")
    
    def main_loop(self):
        """Main processing loop"""
        import threading
        import time
        import cv2
        import numpy as np
        
        last_instruction_time = 0
        frame_count = 0
        
        print("Starting main processing loop...")
        
        while self.is_running:
            try:
                # Capture frame
                frame = self.camera.capture_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process frame if detection is active
                if self.gui.detection_active:
                    # Predict path status
                    result, predictions, status = self.path_detector.predict_path_status(frame)
                    
                    if result:
                        # Update frame size for zone detector
                        height, width = frame.shape[:2]
                        self.zone_detector.update_frame_size(width, height)
                        
                        # Analyze frame zones for obstacles
                        try:
                            zone_analysis = self.zone_detector.analyze_zones(frame, result)
                        except Exception as e:
                            print(f"Zone analysis error: {e}")
                            zone_analysis = {
                                'left': {'blocked': False, 'x_start': 0, 'x_end': 213, 'y_start': 0, 'y_end': 480},
                                'center': {'blocked': False, 'x_start': 213, 'x_end': 426, 'y_start': 0, 'y_end': 480},
                                'right': {'blocked': False, 'x_start': 426, 'x_end': 640, 'y_start': 0, 'y_end': 480}
                            }
                        
                        # Generate directional navigation instruction
                        try:
                            instruction = self.navigation.generate_directional_instruction(result['status'], zone_analysis)
                        except Exception as e:
                            print(f"Navigation error: {e}")
                            instruction = "Proceed with caution"
                        
                        # Update GUI displays
                        self.gui.update_path_status(result['status'], result['confidence'])
                        self.gui.update_instruction(instruction)
                        
                        # Voice guidance
                        if self.gui.voice_active:
                            current_time = time.time()
                            if current_time - last_instruction_time > INSTRUCTION_DELAY:
                                try:
                                    is_emergency = (result['status'] == 'Fully Blocked' and 
                                                  result['confidence'] > 0.7)
                                    self.voice_guide.speak(instruction, priority=is_emergency)
                                except Exception as e:
                                    print(f"Voice error: {e}")
                                last_instruction_time = current_time
                        
                        # Draw overlays
                        try:
                            frame = self.draw_status_overlay(frame, result)
                            frame = self.draw_obstacle_zones(frame, zone_analysis, result)
                        except Exception as e:
                            print(f"Drawing error: {e}")
                
                # Update GUI camera feed
                self.gui.update_camera_feed(frame)
                
                # Update FPS display
                fps = self.camera.get_fps()
                self.gui.update_fps(fps)
                
                frame_count += 1
                
                # Sleep to maintain target FPS
                time.sleep(1.0 / PROCESSING_FPS)
                
            except Exception as e:
                print(f"Main loop error: {e}")
                time.sleep(0.1)
        
        print("Main processing loop ended")
    
    def draw_status_overlay(self, frame, result):
        """Draw status overlay on frame"""
        # Simple status overlay
        status = result['status']
        confidence = result['confidence']
        
        # Draw status text
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def draw_obstacle_zones(self, frame, zone_analysis, result):
        """Draw obstacle zones on frame"""
        height, width = frame.shape[:2]
        
        # Draw zone rectangles
        for zone_name, zone_info in zone_analysis.items():
            if zone_info.get('blocked', False):
                color = (0, 0, 255)  # Red for blocked
            else:
                color = (0, 255, 0)  # Green for clear
            
            x_start = zone_info.get('x_start', 0)
            x_end = zone_info.get('x_end', width)
            y_start = zone_info.get('y_start', 0)
            y_end = zone_info.get('y_end', height)
            
            # Draw rectangle
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
            
            # Add zone label
            cv2.putText(frame, zone_name.upper(), (x_start + 5, y_start + 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up camera application...")
        
        self.is_running = False
        
        if self.camera:
            self.camera.release()
        
        if self.voice_guide:
            self.voice_guide.cleanup()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        print("Camera application cleanup complete")
    
    def run(self):
        """Run the camera application"""
        try:
            print("Starting camera application...")
            
            # Show the GUI
            self.gui.show()
            
            # Run the PyQt application
            exit_code = self.app.exec_()
            
            print(f"Camera application exited with code: {exit_code}")
            return exit_code
            
        except KeyboardInterrupt:
            print("Camera application interrupted")
            return 1
        except Exception as e:
            print(f"Camera application error: {e}")
            return 1
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    try:
        # Create and run the camera application
        app = CameraApp()
        exit_code = app.run()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Failed to start camera application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
