"""
Integrated Assistive Navigation System - Main Application
Combines MobileNet path detection with PyQt5 GUI and voice guidance
"""

import sys
import threading
import time
import cv2
import numpy as np

from PyQt5.QtWidgets import QApplication
from utils.gui import AssistiveGUI
from utils.camera import CameraCapture
from core.path_detector import PathDetector
from core.zone_detector import ZoneDetector
from core.navigation import NavigationAssistant
from utils.voice_guide import VoiceGuide
from config import (VOICE_MESSAGES, SAFETY_COLORS, PROCESSING_FPS, INSTRUCTION_DELAY,
                   VOICE_ENABLED, VOICE_RATE, VOICE_VOLUME)

class IntegratedAssistiveSystem:
    def __init__(self):
        print("Initializing Integrated Assistive Navigation System...")
        
        # Create PyQt application
        self.app = QApplication(sys.argv)
        self.gui = AssistiveGUI()
        
        # Initialize system components
        self.camera = CameraCapture()
        self.path_detector = PathDetector()
        self.zone_detector = ZoneDetector()
        self.navigation = NavigationAssistant()
        self.voice_guide = VoiceGuide()
        
        # System state
        self.running = False
        self.processing_thread = None
        
        # Performance tracking
        self.last_instruction_time = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Connect GUI signals
        self.gui.camera_toggled.connect(self.on_camera_toggled)
        self.gui.detection_toggled.connect(self.on_detection_toggled)
        self.gui.voice_toggled.connect(self.on_voice_toggled)
        self.gui.safety_mode_changed.connect(self.on_safety_mode_changed)
        self.gui.test_voice_requested.connect(self.test_voice)
        
        # Initialize system
        self.initialize_system()
        
    def initialize_system(self):
        """Initialize all system components"""
        try:
            # Check if model is loaded
            if not self.path_detector.is_model_ready():
                self.gui.show_error("Model Error", "Failed to load path detection model")
                return
            
            # Initialize voice guide
            if self.voice_guide.enabled:
                self.gui.update_status("Voice Ready")
                self.voice_guide.speak("Integrated assistive system ready")
            else:
                self.gui.update_status("Voice Disabled")
            
            # Initialize path detector
            self.gui.update_status("Detector Ready")
            
            # Start main processing loop
            self.running = True
            self.processing_thread = threading.Thread(target=self.main_loop, daemon=True)
            self.processing_thread.start()
            
            self.gui.update_status("System Ready")
            print("System initialized successfully")
            
        except Exception as e:
            print(f"System initialization error: {e}")
            self.gui.show_error("Initialization Error", f"Failed to initialize system: {e}")
    
    def main_loop(self):
        """Main processing loop"""
        print("Starting main processing loop...")
        
        while self.running:
            try:
                # Check if camera is active
                if self.gui.camera_active:
                    # Initialize camera if not already done
                    if not self.camera.is_camera_available():
                        if not self.camera.initialize():
                            self.gui.update_status("Camera Error", "red")
                            time.sleep(1)
                            continue
                    
                    # Capture frame
                    frame = self.camera.capture_frame()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    
                    # Process frame if detection is active
                    if self.gui.detection_active:
                        try:
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
                                    # Create default zone analysis
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
                                    if current_time - self.last_instruction_time > INSTRUCTION_DELAY:
                                        # Check for emergency
                                        is_emergency = (result['status'] == 'Fully Blocked' and 
                                                      result['confidence'] > 0.7)
                                        try:
                                            self.voice_guide.speak(instruction, priority=is_emergency)
                                        except Exception as e:
                                            print(f"Voice error: {e}")
                                        self.last_instruction_time = current_time
                                
                                # Draw status overlay and zone analysis on frame
                                try:
                                    frame = self.draw_status_overlay(frame, result)
                                    frame = self.draw_obstacle_zones(frame, zone_analysis, result)
                                except Exception as e:
                                    print(f"Drawing error: {e}")
                                    # Continue with original frame if drawing fails
                            
                        except Exception as e:
                            print(f"Detection processing error: {e}")
                            # Continue with camera feed even if detection fails
                    
                    # Update GUI camera feed (processed or raw frame)
                    self.gui.update_camera_feed(frame)
                    
                    # Update FPS display
                    fps = self.camera.get_fps()
                    self.gui.update_fps(fps)
                    
                    self.frame_count += 1
                
                else:
                    # Camera is not active, release resources
                    if self.camera.is_camera_available():
                        self.camera.release()
                
                # Sleep to maintain target FPS
                time.sleep(1.0 / PROCESSING_FPS)
                
            except Exception as e:
                print(f"Main loop error: {e}")
                self.gui.update_status("Processing Error", "red")
                time.sleep(1)
    
    def generate_instruction(self, result):
        """Generate navigation instruction based on path status"""
        status = result['status']
        confidence = result['confidence']
        
        if status == "Clear":
            if confidence > 0.8:
                return "Path is clear, proceed safely"
            else:
                return "Path appears clear, proceed with caution"
        elif status == "Partially Blocked":
            if confidence > 0.7:
                return "Warning: Obstacle detected, slow down"
            else:
                return "Caution: Partial obstruction ahead"
        elif status == "Fully Blocked":
            if confidence > 0.7:
                return "STOP! Way is blocked"
            else:
                return "Stop: Path blocked ahead"
        else:
            return "Unable to determine path status"
    
    def draw_status_overlay(self, frame, result):
        """Draw status overlay on the frame"""
        if frame is None or result is None:
            return frame
        
        # Create overlay
        overlay = frame.copy()
        height, width = overlay.shape[:2]
        
        # Get status color
        status_color = SAFETY_COLORS.get(result['status'], (128, 128, 128))
        
        # Draw status bar at top
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        
        # Add semi-transparent overlay
        alpha = 0.7
        cv2.addWeighted(overlay[0:80, :], alpha, frame[0:80, :], 1-alpha, 0, frame[0:80, :])
        
        # Draw status text
        status_text = f"STATUS: {result['status'].upper()}"
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Draw confidence and mode
        info_text = f"Confidence: {result['confidence']:.2f} | Mode: {result['safety_mode']}"
        cv2.putText(frame, info_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw probabilities at bottom
        probs = result['probabilities']
        prob_text = f"Clear: {probs['clear']:.2f} | Partial: {probs['partial']:.2f} | Blocked: {probs['full']:.2f}"
        cv2.putText(frame, prob_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_obstacle_zones(self, frame, zone_analysis, result):
        """Draw colored obstacle zones on frame"""
        if frame is None or zone_analysis is None:
            return frame
        
        height, width = frame.shape[:2]
        
        # Define zone colors based on path status
        if result['status'] == "Clear":
            zone_colors = {
                'left': (0, 255, 0),      # Green
                'center': (0, 255, 0),    # Green  
                'right': (0, 255, 0)       # Green
            }
        elif result['status'] == "Partially Blocked":
            zone_colors = {
                'left': (0, 255, 255),    # Yellow
                'center': (0, 165, 255),   # Orange
                'right': (0, 255, 255)    # Yellow
            }
        else:  # Fully Blocked
            zone_colors = {
                'left': (0, 0, 255),      # Red
                'center': (0, 0, 255),     # Red
                'right': (0, 0, 255)       # Red
            }
        
        # Draw zone overlays
        for zone_name, zone_info in zone_analysis.items():
            if zone_name in zone_colors:
                color = zone_colors[zone_name]
                alpha = 0.3
                
                # Create overlay for this zone
                overlay = frame.copy()
                x_start = zone_info['x_start']
                x_end = zone_info['x_end']
                y_start = zone_info['y_start'] 
                y_end = zone_info['y_end']
                
                # Draw semi-transparent rectangle
                cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), color, -1)
                cv2.addWeighted(overlay[y_start:y_end, x_start:x_end], alpha, 
                               frame[y_start:y_end, x_start:x_end], 1-alpha, 0, 
                               frame[y_start:y_end, x_start:x_end])
                
                # Draw zone border
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
                
                # Add zone label
                label = f"{zone_name.upper()}"
                if zone_info.get('blocked', False):
                    label += " [BLOCKED]"
                cv2.putText(frame, label, (x_start + 10, y_start + 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add directional arrow based on instruction
        instruction = self.navigation.generate_directional_instruction(result['status'], zone_analysis)
        if "Go left" in instruction:
            # Draw left arrow
            cv2.arrowedLine(frame, (width//2, height-50), (width//4, height-50), (0, 255, 0), 5)
        elif "Go right" in instruction:
            # Draw right arrow  
            cv2.arrowedLine(frame, (width//2, height-50), (3*width//4, height-50), (0, 255, 0), 5)
        elif "Go straight" in instruction:
            # Draw up arrow
            cv2.arrowedLine(frame, (width//2, height-50), (width//2, height//2), (0, 255, 0), 5)
        elif "Stop" in instruction:
            # Draw stop sign
            center = (width//2, height-50)
            cv2.circle(frame, center, 30, (0, 0, 255), -1)
            cv2.putText(frame, "STOP", (center[0]-25, center[1]+8), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def on_camera_toggled(self, active):
        """Handle camera toggle signal"""
        print(f"Camera toggled: {active}")
        
    def on_detection_toggled(self, active):
        """Handle detection toggle signal"""
        print(f"Detection toggled: {active}")
        
    def on_voice_toggled(self, active):
        """Handle voice toggle signal"""
        print(f"Voice toggled: {active}")
        if active:
            self.voice_guide.enable()
        else:
            self.voice_guide.disable()
        
    def on_safety_mode_changed(self, mode):
        """Handle safety mode change"""
        print(f"Safety mode changed to: {mode}")
        self.path_detector.set_safety_mode(mode)
        
    def test_voice(self):
        """Test voice functionality"""
        test_messages = [
            "Voice test complete",
            "System is ready to use",
            "Path clear",
            "Warning obstacle detected"
        ]
        
        def speak_test():
            for message in test_messages:
                self.voice_guide.speak(message)
                time.sleep(2)
        
        # Run test in separate thread
        test_thread = threading.Thread(target=speak_test, daemon=True)
        test_thread.start()
    
    def cleanup(self):
        """Cleanup system resources"""
        print("Cleaning up system resources...")
        self.running = False
        
        # Stop camera
        if self.camera:
            self.camera.release()
        
        # Stop voice guide
        if self.voice_guide:
            self.voice_guide.cleanup()
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        print("Cleanup complete")
    
    def run(self):
        """Run the application"""
        try:
            print("Starting integrated assistive system...")
            
            # Show the GUI
            self.gui.show()
            
            # Run the PyQt application
            exit_code = self.app.exec_()
            
            print(f"Application exited with code: {exit_code}")
            return exit_code
            
        except KeyboardInterrupt:
            print("Application interrupted by user")
            return 1
        except Exception as e:
            print(f"Application error: {e}")
            self.gui.show_error("Application Error", f"An error occurred: {e}")
            return 1
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    try:
        # Create and run the application
        app = IntegratedAssistiveSystem()
        exit_code = app.run()
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
