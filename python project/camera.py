import cv2
import numpy as np

class CameraCapture:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.frame = None
        self.is_running = False
        
    def initialize(self):
        """Initialize the camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {self.camera_id}")
            
            # Set camera resolution for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_running = True
            return True
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def capture_frame(self):
        """Capture a single frame from the camera"""
        if self.cap is None or not self.is_running:
            return None
            
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            return frame
        else:
            print("Failed to capture frame")
            return None
    
    def get_frame(self):
        """Get the current frame"""
        return self.frame
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
    
    def is_camera_available(self):
        """Check if camera is available and running"""
        return self.cap is not None and self.is_running and self.cap.isOpened()
