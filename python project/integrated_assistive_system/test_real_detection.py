"""
Test Real Object Detection
Tests the enhanced system with real object detection and visual zones
"""

import cv2
import numpy as np
import time
from camera import CameraCapture
from path_detector import PathDetector
from zone_detector import ZoneDetector
from navigation import NavigationAssistant
from voice_guide import VoiceGuide
from real_object_detector import RealObjectDetector

def test_real_object_detection():
    """Test real object detection with camera input"""
    print("=== Testing Real Object Detection ===")
    
    # Initialize components
    camera = CameraCapture()
    path_detector = PathDetector()
    zone_detector = ZoneDetector()
    navigation = NavigationAssistant()
    voice_guide = VoiceGuide()
    object_detector = RealObjectDetector()
    
    # Check if model loaded successfully
    if not path_detector.is_model_ready():
        print("❌ Model failed to load")
        return False
    
    print("✅ Model loaded successfully")
    print("✅ Object detector initialized")
    
    # Initialize camera
    if not camera.initialize():
        print("❌ Camera failed to initialize")
        return False
    
    print("✅ Camera initialized")
    print("✅ Voice guide ready")
    
    print("\n=== Starting Real Object Detection Test ===")
    print("Press 'q' to quit, 'm' to change safety mode")
    print("Watch for:")
    print("  • Colored bounding boxes around detected objects")
    print("  • Zone lines (LEFT/CENTER/RIGHT)")
    print("  • Directional instructions: 'Go left', 'Go right', 'Go straight'")
    print("  • Visual status overlay")
    
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
            path_detector.set_safety_mode(current_mode)
            
            # Predict path status
            result, predictions, status = path_detector.predict_path_status(frame)
            
            if result:
                frame_count += 1
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Update frame size for zone detector
                height, width = frame.shape[:2]
                zone_detector.update_frame_size(width, height)
                
                # Detect real objects in the frame
                detected_objects = object_detector.detect_objects(frame)
                navigation_objects = object_detector.filter_navigation_objects(detected_objects)
                
                # Categorize objects into zones
                objects_in_zones = zone_detector.categorize_objects(navigation_objects)
                
                # Generate directional navigation instruction
                instruction = navigation.generate_directional_instruction(result['status'], objects_in_zones)
                
                # Voice guidance (every 3 seconds)
                current_time = time.time()
                if current_time - last_voice_time > 3.0:
                    voice_guide.speak(instruction)
                    last_voice_time = current_time
                    print(f"🔊 VOICE: {instruction}")
                
                # Display results
                obj_count = len(navigation_objects)
                left_count = len(objects_in_zones['left'])
                center_count = len(objects_in_zones['center'])
                right_count = len(objects_in_zones['right'])
                
                print(f"Frame {frame_count:4d} | FPS: {fps:5.1f} | Mode: {current_mode:12s} | "
                      f"Path: {status:17s} | Instruction: {instruction:25s} | "
                      f"Objects: {obj_count} (L:{left_count} C:{center_count} R:{right_count})")
                
                # Draw all overlays on frame
                # 1. Status overlay
                color = path_detector.get_safety_color(status)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
                overlay = frame.copy()
                cv2.addWeighted(overlay[0:80, :], 0.7, frame[0:80, :], 0.3, 0, frame[0:80, :])
                
                cv2.putText(frame, f"PATH: {status.upper()}", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"INSTRUCTION: {instruction.upper()}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Mode: {current_mode} | Conf: {result['confidence']:.2f}", 
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # 2. Zone lines
                frame = zone_detector.draw_zones(frame)
                
                # 3. Detected objects with bounding boxes
                frame = object_detector.draw_objects(frame, navigation_objects)
                
                # Show frame
                cv2.imshow('Real Object Detection Test', frame)
                
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
                voice_guide.speak("Testing real object detection")
                print("🔊 VOICE: Testing real object detection")
            elif key == ord('s'):
                # Save screenshot
                screenshot_name = f"object_detection_frame_{frame_count}.jpg"
                cv2.imwrite(screenshot_name, frame)
                print(f"📸 Screenshot saved: {screenshot_name}")
            
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

def test_object_detector_only():
    """Test object detector without camera"""
    print("=== Testing Object Detector Only ===")
    
    detector = RealObjectDetector()
    
    # Create test image with some shapes
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some test objects (rectangles)
    cv2.rectangle(test_image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(test_image, (300, 150), (400, 300), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(test_image, (450, 200), (550, 350), (0, 0, 255), -1)  # Red rectangle
    
    print("Testing with synthetic image containing 3 rectangles...")
    
    # Detect objects
    objects = detector.detect_objects(test_image)
    navigation_objects = detector.filter_navigation_objects(objects)
    
    print(f"Total objects detected: {len(objects)}")
    print(f"Navigation-relevant objects: {len(navigation_objects)}")
    
    for i, obj in enumerate(navigation_objects):
        print(f"  Object {i+1}: {obj['class']} (conf: {obj['confidence']:.2f}, area: {obj['area']})")
    
    # Draw objects and show
    result_image = detector.draw_objects(test_image, navigation_objects)
    cv2.imshow('Object Detector Test', result_image)
    cv2.waitKey(3000)  # Show for 3 seconds
    cv2.destroyAllWindows()
    
    print("✅ Object detector test complete")
    return True

if __name__ == "__main__":
    print("Real Object Detection Test")
    print("==========================")
    
    # First test object detector only
    if test_object_detector_only():
        print("\n" + "="*50)
        
        # Ask user if they want to test with camera
        try:
            response = input("\nDo you want to test with live camera? (y/n): ").lower()
            if response in ['y', 'yes']:
                test_real_object_detection()
            else:
                print("Skipping camera test")
        except KeyboardInterrupt:
            print("\nTest cancelled")
    else:
        print("❌ Object detector test failed, skipping camera test")
