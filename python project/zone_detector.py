import cv2
import numpy as np

class ZoneDetector:
    def __init__(self, frame_width=640, frame_height=480):
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
    
    def get_all_closest_objects(self):
        """Get the closest object in each zone"""
        closest_objects = {}
        for zone_name in ['left', 'center', 'right']:
            closest_obj = self.get_closest_object(zone_name)
            if closest_obj:
                closest_objects[zone_name] = closest_obj
        return closest_objects
    
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
        
        return frame_copy
    
    def has_objects_in_zone(self, zone_name):
        """Check if there are any objects in a specific zone"""
        return len(self.objects_in_zones.get(zone_name, [])) > 0
    
    def get_zone_status(self):
        """Get status of all zones (whether they contain objects)"""
        status = {}
        for zone_name in ['left', 'center', 'right']:
            status[zone_name] = self.has_objects_in_zone(zone_name)
        return status
    
    def get_danger_zones(self):
        """Identify zones with potential obstacles"""
        danger_zones = []
        for zone_name in ['left', 'center', 'right']:
            if self.has_objects_in_zone(zone_name):
                closest_obj = self.get_closest_object(zone_name)
                if closest_obj and closest_obj['area'] > 10000:  # Threshold for "close" objects
                    danger_zones.append(zone_name)
        return danger_zones
