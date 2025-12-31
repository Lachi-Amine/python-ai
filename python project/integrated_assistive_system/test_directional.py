"""
Test script for directional navigation
Tests the zone detection and directional instructions
"""

import cv2
import numpy as np
import time
import random
from path_detector import PathDetector
from zone_detector import ZoneDetector
from navigation import NavigationAssistant
from voice_guide import VoiceGuide
from camera import CameraCapture

def test_directional_navigation():
    """Test directional navigation with camera input"""
    print("=== Testing Directional Navigation ===")
    
    # Initialize components
    detector = PathDetector()
    zone_detector = ZoneDetector()
    navigation = NavigationAssistant()
    voice_guide = VoiceGuide()
    camera = CameraCapture()
    
    # Check if model loaded successfully
    if not detector.is_model_ready():
        print("❌ Model failed to load")
        return False
    
    print("✅ Model loaded successfully")
    
    # Initialize camera
    if not camera.initialize():
        print("❌ Camera failed to initialize")
        return False
    
    print("✅ Camera initialized")
    print("✅ Voice guide ready")
    
    print("\n=== Starting Directional Navigation Test ===")
    print("Press 'q' to quit, 'm' to change safety mode")
    print("Listen for directional instructions: 'Go left', 'Go right', 'Go straight', etc.")
    
    modes = ["conservative", "balanced", "aggressive"]
    current_mode_idx = 0
    frame_count = 0
    start_time = time.time()
    last_voice_time = 0
    
    try:
        while True:
            # Capture frame
            frame = camera.capture_frame()
            if frame is None:
                print("⚠️  Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Set current safety mode
            current_mode = modes[current_mode_idx]
            detector.set_safety_mode(current_mode)
            
            # Predict path status
            result, predictions, status = detector.predict_path_status(frame)
            
            if result:
                frame_count += 1
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Update frame size for zone detector
                height, width = frame.shape[:2]
                zone_detector.update_frame_size(width, height)
                
                # Simulate object detection based on path status
                objects_in_zones = zone_detector.simulate_object_detection(frame, result['status'])
                
                # Generate directional navigation instruction
                instruction = navigation.generate_directional_instruction(result['status'], objects_in_zones)
                
                # Voice guidance (every 3 seconds)
                current_time = time.time()
                if current_time - last_voice_time > 3.0:
                    voice_guide.speak(instruction)
                    last_voice_time = current_time
                    print(f"🔊 VOICE: {instruction}")
                
                # Display results
                print(f"Frame {frame_count:4d} | FPS: {fps:5.1f} | Mode: {current_mode:12s} | "
                      f"Path: {status:17s} | Instruction: {instruction:25s} | "
                      f"Zones: L:{len(objects_in_zones['left'])} C:{len(objects_in_zones['center'])} R:{len(objects_in_zones['right'])}")
                
                # Draw status overlay and zones on frame
                color = detector.get_safety_color(status)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
                
                # Add semi-transparent overlay
                overlay = frame.copy()
                cv2.addWeighted(overlay[0:80, :], 0.7, frame[0:80, :], 0.3, 0, frame[0:80, :])
                
                # Draw text
                cv2.putText(frame, f"PATH: {status.upper()}", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"INSTRUCTION: {instruction.upper()}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Mode: {current_mode} | Conf: {result['confidence']:.2f}", 
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Draw zones and objects
                frame = zone_detector.draw_zones(frame)
                
                # Show frame
                cv2.imshow('Directional Navigation Test', frame)
                
            else:
                print("⚠️  Prediction failed")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                current_mode_idx = (current_mode_idx + 1) % len(modes)
                print(f"🔄 Switched to {modes[current_mode_idx]} mode")
            elif key == ord('v'):
                # Manual voice test
                voice_guide.speak("Testing directional navigation")
                print("🔊 VOICE: Testing directional navigation")
            
            # Small delay to prevent overwhelming
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        return False
    
    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        voice_guide.cleanup()
        print("\n=== Test Complete ===")
        print(f"Total frames processed: {frame_count}")
        if frame_count > 0:
            avg_fps = frame_count / (time.time() - start_time)
            print(f"Average FPS: {avg_fps:.1f}")
    
    return True

def test_directional_logic_only():
    """Test directional navigation logic without camera"""
    print("=== Testing Directional Navigation Logic ===")
    
    detector = PathDetector()
    zone_detector = ZoneDetector()
    navigation = NavigationAssistant()
    voice_guide = VoiceGuide()
    
    if not detector.is_model_ready():
        print("❌ Model failed to load")
        return False
    
    print("✅ Model loaded successfully")
    
    # Create dummy scenarios
    scenarios = [
        {
            'name': 'Clear Path',
            'path_status': 'Clear',
            'left_objects': [],
            'center_objects': [],
            'right_objects': []
        },
        {
            'name': 'Left Obstacle',
            'path_status': 'Partially Blocked',
            'left_objects': [{'class': 'obstacle', 'area': 12000}],
            'center_objects': [],
            'right_objects': []
        },
        {
            'name': 'Right Obstacle',
            'path_status': 'Partially Blocked',
            'left_objects': [],
            'center_objects': [],
            'right_objects': [{'class': 'obstacle', 'area': 12000}]
        },
        {
            'name': 'Center Obstacle',
            'path_status': 'Fully Blocked',
            'left_objects': [],
            'center_objects': [{'class': 'obstacle', 'area': 20000}],
            'right_objects': []
        },
        {
            'name': 'Both Sides Blocked',
            'path_status': 'Partially Blocked',
            'left_objects': [{'class': 'obstacle', 'area': 10000}],
            'center_objects': [],
            'right_objects': [{'class': 'obstacle', 'area': 10000}]
        }
    ]
    
    print("\nTesting different scenarios:")
    print("-" * 60)
    
    for scenario in scenarios:
        objects_in_zones = {
            'left': scenario['left_objects'],
            'center': scenario['center_objects'],
            'right': scenario['right_objects']
        }
        
        instruction = navigation.generate_directional_instruction(
            scenario['path_status'], objects_in_zones
        )
        
        print(f"Scenario: {scenario['name']:25s} | Instruction: {instruction}")
        
        # Test voice
        voice_guide.speak(instruction)
        time.sleep(2)  # Wait for voice to complete
    
    print("✅ Directional logic test complete")
    return True

if __name__ == "__main__":
    print("Directional Navigation Test")
    print("==========================")
    
    # First test logic only
    if test_directional_logic_only():
        print("\n" + "="*50)
        
        # Ask user if they want to test with camera
        try:
            response = input("\nDo you want to test with live camera? (y/n): ").lower()
            if response in ['y', 'yes']:
                test_directional_navigation()
            else:
                print("Skipping camera test")
        except KeyboardInterrupt:
            print("\nTest cancelled")
    else:
        print("❌ Logic test failed, skipping camera test")
