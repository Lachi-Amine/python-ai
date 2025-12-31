import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = None
        self.model_path = model_path
        self.detected_objects = []
        self.confidence_threshold = 0.5
        
        # Classes relevant for navigation assistance
        self.navigation_classes = {
            'person', 'chair', 'table', 'door', 'bicycle', 'car', 'motorcycle',
            'bus', 'truck', 'traffic light', 'stop sign', 'parking meter',
            'bench', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'couch', 'potted plant', 'bed', 'mirror', 'dining table',
            'window', 'desk', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        }
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"YOLO model loaded successfully: {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False
    
    def detect_objects(self, frame):
        """Detect objects in the given frame"""
        if self.model is None:
            print("Model not loaded")
            return []
        
        try:
            # Run inference
            results = self.model(frame, verbose=False)
            
            self.detected_objects = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get confidence score
                        confidence = float(box.conf[0])
                        
                        # Filter by confidence threshold
                        if confidence >= self.confidence_threshold:
                            # Get class name
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]
                            
                            # Only consider navigation-relevant objects
                            if class_name in self.navigation_classes:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                
                                # Calculate center point
                                center_x = int((x1 + x2) / 2)
                                center_y = int((y1 + y2) / 2)
                                
                                # Store detected object
                                obj_info = {
                                    'class': class_name,
                                    'confidence': confidence,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'center': [center_x, center_y],
                                    'area': int((x2 - x1) * (y2 - y1))
                                }
                                self.detected_objects.append(obj_info)
            
            return self.detected_objects
            
        except Exception as e:
            print(f"Error during object detection: {e}")
            return []
    
    def get_detected_objects(self):
        """Get the list of detected objects"""
        return self.detected_objects
    
    def draw_detections(self, frame):
        """Draw bounding boxes and labels on the frame"""
        if frame is None:
            return frame
            
        frame_copy = frame.copy()
        
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['bbox']
            class_name = obj['class']
            confidence = obj['confidence']
            center_x, center_y = obj['center']
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(frame_copy, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame_copy
    
    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold for detection"""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
        else:
            print("Confidence threshold must be between 0.0 and 1.0")
