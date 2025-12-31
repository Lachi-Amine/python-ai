import sys
import threading
import time

# Import PyQt5 for GUI
from PyQt5.QtWidgets import QApplication
from pyqt_gui import AssistiveGUI

# Import custom modules
from camera import CameraCapture
from simple_detector import SimpleObjectDetector as ObjectDetector
from zone_detector import ZoneDetector
from navigation import NavigationAssistant
from voice_guide import VoiceGuide

class TestAssistiveSystem:
    def __init__(self):
        # Create PyQt application
        self.app = QApplication(sys.argv)
        self.gui = AssistiveGUI()
        
        # Initialize system components
        self.camera = CameraCapture()
        self.object_detector = ObjectDetector()
        self.zone_detector = ZoneDetector()
        self.navigation = NavigationAssistant()
        self.voice_guide = VoiceGuide()
        
        # System state
        self.running = False
        self.processing_thread = None
        
        # Connect GUI signals
        self.gui.camera_toggled.connect(self.on_camera_toggled)
        self.gui.detection_toggled.connect(self.on_detection_toggled)
        self.gui.voice_toggled.connect(self.on_voice_toggled)
        self.gui.test_voice_requested.connect(self.test_voice)
        
        # Initialize components
        self.initialize_system()
        
    def initialize_system(self):
        """Initialize all system components"""
        try:
            # Initialize voice guide
            if self.voice_guide.initialize():
                self.gui.update_status("Voice Ready")
                self.voice_guide.speak("Assistive navigation system ready")
            else:
                self.gui.update_status("Voice Error")
                self.gui.show_error("Voice Error", "Failed to initialize voice guide")
            
            # Initialize simple object detector
            self.gui.update_status("Detector Ready")
            
            # Start main processing loop
            self.running = True
            self.processing_thread = threading.Thread(target=self.main_loop, daemon=True)
            self.processing_thread.start()
            
        except Exception as e:
            print(f"System initialization error: {e}")
            self.gui.show_error("Initialization Error", f"Failed to initialize system: {e}")
    
    def main_loop(self):
        """Main processing loop"""
        frame_count = 0
        last_instruction_time = 0
        
        while self.running:
            try:
                # Check if camera is active
                if self.gui.camera_active:
                    # Initialize camera if not already done
                    if not self.camera.is_camera_available():
                        if not self.camera.initialize():
                            self.gui.update_status("Camera Error")
                            time.sleep(1)
                            continue
                    
                    # Capture frame
                    frame = self.camera.capture_frame()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    
                    # Update frame size for zone detector
                    height, width = frame.shape[:2]
                    self.zone_detector.update_frame_size(width, height)
                    
                    # Process frame if detection is active
                    if self.gui.detection_active:
                        print("Processing frame...")
                        
                        # Detect objects
                        detected_objects = self.object_detector.detect_objects(frame)
                        print(f"Detected {len(detected_objects)} objects")
                        
                        # Update confidence threshold from GUI
                        confidence = self.gui.get_confidence_threshold()
                        self.object_detector.set_confidence_threshold(confidence)
                        
                        # Categorize objects into zones
                        objects_in_zones = self.zone_detector.categorize_objects(detected_objects)
                        print(f"Zone categorization: {objects_in_zones}")
                        
                        # Generate navigation instruction
                        instruction = self.navigation.generate_instruction(objects_in_zones)
                        print(f"Instruction: {instruction}")
                        
                        # Update GUI displays
                        self.gui.update_instruction(instruction)
                        self.gui.update_object_count(len(detected_objects))
                        self.gui.update_zone_status(self.zone_detector.get_zone_status())
                        
                        # Voice guidance
                        if self.gui.voice_active and instruction:
                            current_time = time.time()
                            if current_time - last_instruction_time > 3.0:  # Speak every 3 seconds
                                # Check for emergency
                                is_emergency = self.navigation.is_emergency_stop(objects_in_zones)
                                self.voice_guide.speak_instruction(instruction, is_emergency)
                                last_instruction_time = current_time
                        
                        # Draw detections and zones on frame
                        frame = self.object_detector.draw_detections(frame)
                        frame = self.zone_detector.draw_zones(frame)
                    
                    # Update camera feed display
                    self.gui.update_camera_feed(frame)
                    
                    frame_count += 1
                    
                    # Update voice rate from GUI
                    voice_rate = self.gui.get_voice_rate()
                    self.voice_guide.set_voice_rate(voice_rate)
                
                else:
                    # Camera is not active, release resources
                    if self.camera.is_camera_available():
                        self.camera.release()
                
                # Sleep to prevent excessive CPU usage
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"Main loop error: {e}")
                import traceback
                traceback.print_exc()
                self.gui.update_status("Processing Error")
                time.sleep(1)
    
    def on_camera_toggled(self, active):
        """Handle camera toggle signal"""
        pass  # Camera state is handled by GUI
        
    def on_detection_toggled(self, active):
        """Handle detection toggle signal"""
        pass  # Detection state is handled by GUI
        
    def on_voice_toggled(self, active):
        """Handle voice toggle signal"""
        pass  # Voice state is handled by GUI

    def test_voice(self):
        """Test voice functionality"""
        test_messages = [
            "Voice test complete",
            "System is ready to use"
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
    
    def run(self):
        """Run the application"""
        try:
            # Start the PyQt application
            self.gui.show()
            self.app.exec_()
            
        except KeyboardInterrupt:
            print("Application interrupted")
        except Exception as e:
            print(f"Application error: {e}")
            self.gui.show_error("Application Error", f"An error occurred: {e}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    try:
        # Create and run the application
        app = TestAssistiveSystem()
        app.run()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
