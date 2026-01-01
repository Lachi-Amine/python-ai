"""
Zone Detection Module
Divides camera view into zones and provides directional navigation
"""

import cv2
import numpy as np
from config import CAMERA_WIDTH, CAMERA_HEIGHT

class ZoneDetector:
    def __init__(self, frame_width=CAMERA_WIDTH, frame_height=CAMERA_HEIGHT):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.zones = {
            'left': {'x_start': 0, 'x_end': frame_width // 3, 'y_start': 0, 'y_end': frame_height},
            'center': {'x_start': frame_width // 3, 'x_end': 2 * (frame_width // 3), 'y_start': 0, 'y_end': frame_height},
            'right': {'x_start': 2 * (frame_width // 3), 'x_end': frame_width, 'y_start': 0, 'y_end': frame_height}
        }
        self.objects_in_zones = {'left': [], 'center': [], 'right': []}
        
    def update_frame_size(self, frame_width, frame_height):
        """Update frame dimensions and recalculate zones"""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.zones = {
            'left': {'x_start': 0, 'x_end': frame_width // 3, 'y_start': 0, 'y_end': frame_height},
            'center': {'x_start': frame_width // 3, 'x_end': 2 * (frame_width // 3), 'y_start': 0, 'y_end': frame_height},
            'right': {'x_start': 2 * (frame_width // 3), 'x_end': frame_width, 'y_start': 0, 'y_end': frame_height}
        }
    
    def get_zone(self, x, y):
        """Determine which zone a point (x, y) belongs to"""
        for zone_name, zone_coords in self.zones.items():
            if (zone_coords['x_start'] <= x <= zone_coords['x_end'] and
                zone_coords['y_start'] <= y <= zone_coords['y_end']):
                return zone_name
        return None
    
    def simulate_object_detection(self, frame, path_status):
        """
        Simulate object detection based on path status
        Creates pseudo-objects for navigation guidance
        """
        self.objects_in_zones = {'left': [], 'center': [], 'right': []}
        
        height, width = frame.shape[:2]
        
        if path_status == "Clear":
            # No obstacles detected
            pass
        elif path_status == "Partially Blocked":
            # Add some obstacles to side zones
            import random
            
            # Randomly place obstacles in left or right zone
            if random.random() > 0.5:
                # Obstacle in left zone
                obj = {
                    'class': 'obstacle',
                    'confidence': 0.7,
                    'bbox': [50, height//3, 150, 2*height//3],
                    'center': [100, height//2],
                    'area': 10000
                }
                self.objects_in_zones['left'].append(obj)
            else:
                # Obstacle in right zone
                obj = {
                    'class': 'obstacle', 
                    'confidence': 0.7,
                    'bbox': [width-150, height//3, width-50, 2*height//3],
                    'center': [width-100, height//2],
                    'area': 10000
                }
                self.objects_in_zones['right'].append(obj)
                
        elif path_status == "Fully Blocked":
            # Major obstacle in center zone
            obj = {
                'class': 'obstacle',
                'confidence': 0.9,
                'bbox': [width//3, height//3, 2*width//3, 2*height//3],
                'center': [width//2, height//2],
                'area': 20000
            }
            self.objects_in_zones['center'].append(obj)
            
            # Sometimes add side obstacles too
            import random
            if random.random() > 0.3:
                left_obj = {
                    'class': 'obstacle',
                    'confidence': 0.6,
                    'bbox': [30, height//4, 120, 3*height//4],
                    'center': [75, height//2],
                    'area': 8000
                }
                self.objects_in_zones['left'].append(left_obj)
        
        return self.objects_in_zones
    
    def categorize_objects(self, detected_objects):
        """Categorize detected objects into zones"""
        self.objects_in_zones = {'left': [], 'center': [], 'right': []}
        
        for obj in detected_objects:
            center_x, center_y = obj['center']
            zone = self.get_zone(center_x, center_y)
            if zone:
                self.objects_in_zones[zone].append(obj)
        
        return self.objects_in_zones
    
    def get_objects_in_zone(self, zone_name):
        """Get objects in a specific zone"""
        return self.objects_in_zones.get(zone_name, [])
    
    def get_closest_object(self, zone_name):
        """Get the closest object in a specific zone based on area"""
        objects = self.objects_in_zones.get(zone_name, [])
        if not objects:
            return None
        
        # Sort by area (larger area = closer object)
        closest_obj = max(objects, key=lambda obj: obj['area'])
        return closest_obj
    
    def get_zone_status(self):
        """Get status of all zones (whether they contain objects)"""
        status = {}
        for zone_name in ['left', 'center', 'right']:
            status[zone_name] = len(self.objects_in_zones.get(zone_name, [])) > 0
        return status
    
    def draw_zones(self, frame):
        """Draw zone boundaries on the frame"""
        if frame is None:
            return frame
            
        frame_copy = frame.copy()
        
        # Draw vertical lines to separate zones
        cv2.line(frame_copy, (self.frame_width // 3, 0), 
                (self.frame_width // 3, self.frame_height), (255, 255, 0), 2)
        cv2.line(frame_copy, (2 * (self.frame_width // 3), 0), 
                (2 * (self.frame_width // 3), self.frame_height), (255, 255, 0), 2)
        
        # Add zone labels
        cv2.putText(frame_copy, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 0), 2)
        cv2.putText(frame_copy, "CENTER", (self.frame_width // 3 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame_copy, "RIGHT", (2 * (self.frame_width // 3) + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw detected objects
        for zone_name, objects in self.objects_in_zones.items():
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                class_name = obj['class']
                confidence = obj['confidence']
                
                # Draw bounding box
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame_copy, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame_copy
