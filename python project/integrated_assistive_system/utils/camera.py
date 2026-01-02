"""
Camera Capture Module
Simplified camera interface for the integrated system
"""

import cv2
import numpy as np
from config import CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

class CameraCapture:
    def __init__(self, camera_id=CAMERA_ID):
        self.camera_id = camera_id
        self.cap = None
        self.frame = None
        self.is_running = False
        self.frame_count = 0
        self.start_time = None
        
    def initialize(self):
        """Initialize the camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {self.camera_id}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            
            # Warm-up camera - capture and discard first few frames
            print("Warming up camera...")
            for i in range(3):
                ret, frame = self.cap.read()
                if not ret:
                    print(f"Warning: Failed to capture warm-up frame {i+1}")
                else:
                    print(f"Warm-up frame {i+1} captured successfully")
            
            self.is_running = True
            self.start_time = cv2.getTickCount()
            print(f"Camera {self.camera_id} initialized successfully")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def capture_frame(self):
        """Capture a single frame from the camera"""
        if self.cap is None or not self.is_running:
            return None
        
        try:
            # Try to capture frame with retries
            for attempt in range(3):
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                    self.frame_count += 1
                    return frame
                else:
                    print(f"Frame capture attempt {attempt + 1} failed, retrying...")
                    # Small delay between retries
                    cv2.waitKey(50)
            
            print("Failed to capture frame after 3 attempts")
            return None
                
        except Exception as e:
            print(f"Frame capture error: {e}")
            return None
    
    def get_frame(self):
        """Get the current frame"""
        return self.frame
    
    def get_fps(self):
        """Calculate current FPS"""
        if self.start_time is None or self.frame_count == 0:
            return 0.0
        
        current_time = cv2.getTickCount()
        elapsed_time = (current_time - self.start_time) / cv2.getTickFrequency()
        
        if elapsed_time > 0:
            return self.frame_count / elapsed_time
        return 0.0
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            print("Camera released")
    
    def is_camera_available(self):
        """Check if camera is available and running"""
        return self.cap is not None and self.is_running and self.cap.isOpened()
    
    def get_camera_info(self):
        """Get camera information"""
        if self.cap is None:
            return {}
        
        return {
            'camera_id': self.camera_id,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'current_fps': self.get_fps(),
            'frame_count': self.frame_count
        }
