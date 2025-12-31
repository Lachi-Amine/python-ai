import time
from collections import deque

class NavigationAssistant:
    def __init__(self):
        self.instruction_history = deque(maxlen=5)  # Store last 5 instructions
        self.last_instruction_time = 0
        self.instruction_delay = 3.0  # Minimum seconds between instructions
        self.current_instruction = ""
        self.priority_objects = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']
        
    def generate_instruction(self, objects_in_zones, danger_zones=None):
        """Generate navigation instruction based on detected objects"""
        current_time = time.time()
        
        # Check if enough time has passed since last instruction
        if current_time - self.last_instruction_time < self.instruction_delay:
            return self.current_instruction
        
        # Get zone status
        left_objects = objects_in_zones.get('left', [])
        center_objects = objects_in_zones.get('center', [])
        right_objects = objects_in_zones.get('right', [])
        
        instruction = ""
        
        # Priority 1: Check for immediate obstacles in center (most critical)
        if center_objects:
            closest_center = max(center_objects, key=lambda obj: obj['area'])
            if closest_center['area'] > 15000:  # Large object in center
                instruction = self._generate_center_obstacle_instruction(closest_center)
        
        # Priority 2: Check for obstacles in left and right
        elif left_objects and right_objects:
            # Objects on both sides - suggest going straight if center is clear
            instruction = "Go straight"
        
        elif left_objects:
            closest_left = max(left_objects, key=lambda obj: obj['area'])
            if closest_left['area'] > 10000:
                instruction = "Go right"
            else:
                instruction = "Go straight"
        
        elif right_objects:
            closest_right = max(right_objects, key=lambda obj: obj['area'])
            if closest_right['area'] > 10000:
                instruction = "Go left"
            else:
                instruction = "Go straight"
        
        # Priority 3: No obstacles detected
        else:
            instruction = "Path clear"
        
        # Avoid repeating the same instruction
        if instruction == self.current_instruction:
            return self.current_instruction
        
        # Update instruction history and current instruction
        self.instruction_history.append(instruction)
        self.current_instruction = instruction
        self.last_instruction_time = current_time
        
        return instruction
    
    def _generate_center_obstacle_instruction(self, obj):
        """Generate specific instruction for center obstacle"""
        obj_class = obj['class']
        area = obj['area']
        
        # Check for priority objects that require immediate attention
        if obj_class in self.priority_objects:
            if area > 20000:
                return f"Warning: {obj_class} ahead, stop"
            else:
                return f"{obj_class} ahead, go left or right"
        
        # General obstacle instructions based on size
        if area > 25000:
            return "Large obstacle ahead, go left or right"
        elif area > 15000:
            return "Object ahead, go left or right"
        else:
            return "Object ahead, go left or right"
    
    def get_detailed_instruction(self, objects_in_zones):
        """Generate more detailed navigation instruction"""
        left_objects = objects_in_zones.get('left', [])
        center_objects = objects_in_zones.get('center', [])
        right_objects = objects_in_zones.get('right', [])
        
        details = []
        
        # Describe objects in each zone
        if center_objects:
            closest_center = max(center_objects, key=lambda obj: obj['area'])
            details.append(f"{closest_center['class']} ahead")
        
        if left_objects:
            closest_left = max(left_objects, key=lambda obj: obj['area'])
            details.append(f"{closest_left['class']} on left")
        
        if right_objects:
            closest_right = max(right_objects, key=lambda obj: obj['area'])
            details.append(f"{closest_right['class']} on right")
        
        if not details:
            return "No obstacles detected"
        
        return ". ".join(details)
    
    def get_instruction_history(self):
        """Get the history of recent instructions"""
        return list(self.instruction_history)
    
    def clear_instruction_history(self):
        """Clear the instruction history"""
        self.instruction_history.clear()
        self.current_instruction = ""
        self.last_instruction_time = 0
    
    def set_instruction_delay(self, delay):
        """Set the minimum delay between instructions"""
        if delay > 0:
            self.instruction_delay = delay
    
    def is_emergency_stop(self, objects_in_zones):
        """Check if emergency stop is needed"""
        center_objects = objects_in_zones.get('center', [])
        
        if center_objects:
            closest_center = max(center_objects, key=lambda obj: obj['area'])
            # Very large object in center or priority vehicle
            if (closest_center['area'] > 30000 or 
                closest_center['class'] in ['car', 'bus', 'truck', 'motorcycle'] and 
                closest_center['area'] > 20000):
                return True
        return False
    
    def get_safety_advice(self, objects_in_zones):
        """Get safety advice based on current situation"""
        advice = []
        
        # Check for multiple obstacles
        zones_with_objects = [zone for zone, objects in objects_in_zones.items() if objects]
        
        if len(zones_with_objects) >= 2:
            advice.append("Multiple obstacles detected, proceed with caution")
        
        # Check for moving objects (vehicles, people)
        for zone, objects in objects_in_zones.items():
            for obj in objects:
                if obj['class'] in ['person', 'car', 'bicycle', 'motorcycle']:
                    advice.append(f"Moving {obj['class']} detected in {zone}")
        
        if not advice:
            advice.append("Safe to proceed")
        
        return advice
