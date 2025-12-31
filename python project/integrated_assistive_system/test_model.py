"""
Simple test script to verify the MobileNet model is working
Tests the path detector with a basic camera feed
"""

import cv2
import numpy as np
import time
from path_detector import PathDetector
from camera import CameraCapture

def test_model():
    """Test the path detection model with camera input"""
    print("=== Testing MobileNet Path Detection Model ===")
    
    # Initialize components
    detector = PathDetector()
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
    
    # Test different safety modes
    modes = ["conservative", "balanced", "aggressive"]
    
    print("\n=== Starting Live Test ===")
    print("Press 'q' to quit, 'm' to change safety mode")
    print("Watch the console for real-time predictions...")
    
    current_mode_idx = 0
    frame_count = 0
    start_time = time.time()
    
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
                
                # Display results
                print(f"Frame {frame_count:4d} | FPS: {fps:5.1f} | Mode: {current_mode:12s} | "
                      f"Status: {status:17s} | Confidence: {result['confidence']:.3f} | "
                      f"Probs: C:{result['probabilities']['clear']:.2f} "
                      f"P:{result['probabilities']['partial']:.2f} "
                      f"F:{result['probabilities']['full']:.2f}")
                
                # Draw status on frame
                color = detector.get_safety_color(status)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
                
                # Add semi-transparent overlay
                overlay = frame.copy()
                cv2.addWeighted(overlay[0:60, :], 0.7, frame[0:60, :], 0.3, 0, frame[0:60, :])
                
                # Draw text
                cv2.putText(frame, f"STATUS: {status.upper()}", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Mode: {current_mode} | Conf: {result['confidence']:.2f}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Path Detection Test', frame)
                
            else:
                print("⚠️  Prediction failed")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                current_mode_idx = (current_mode_idx + 1) % len(modes)
                print(f"🔄 Switched to {modes[current_mode_idx]} mode")
            
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
        print("\n=== Test Complete ===")
        print(f"Total frames processed: {frame_count}")
        if frame_count > 0:
            avg_fps = frame_count / (time.time() - start_time)
            print(f"Average FPS: {avg_fps:.1f}")
    
    return True

def test_model_with_dummy_data():
    """Test the model with dummy data (no camera required)"""
    print("=== Testing Model with Dummy Data ===")
    
    detector = PathDetector()
    
    if not detector.is_model_ready():
        print("❌ Model failed to load")
        return False
    
    print("✅ Model loaded successfully")
    
    # Create dummy image (224x224x3 random noise)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    print("\nTesting with dummy image data...")
    
    for mode in ["conservative", "balanced", "aggressive"]:
        detector.set_safety_mode(mode)
        
        # Test multiple times
        results = []
        for i in range(5):
            result, predictions, status = detector.predict_path_status(dummy_image)
            if result:
                results.append(result['status'])
        
        if results:
            most_common = max(set(results), key=results.count)
            print(f"Mode: {mode:12s} | Most common result: {most_common:17s} | "
                  f"Results: {', '.join(results)}")
        else:
            print(f"Mode: {mode:12s} | ❌ No successful predictions")
    
    print("✅ Dummy data test complete")
    return True

if __name__ == "__main__":
    print("MobileNet Path Detection Model Test")
    print("====================================")
    
    # First test with dummy data (no camera needed)
    if test_model_with_dummy_data():
        print("\n" + "="*50)
        
        # Ask user if they want to test with camera
        try:
            response = input("\nDo you want to test with live camera? (y/n): ").lower()
            if response in ['y', 'yes']:
                test_model()
            else:
                print("Skipping camera test")
        except KeyboardInterrupt:
            print("\nTest cancelled")
    else:
        print("❌ Dummy data test failed, skipping camera test")
