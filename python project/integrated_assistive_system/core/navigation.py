"""
Navigation Assistant Module
Provides directional navigation instructions based on zone analysis
"""

import time
from collections import deque

class NavigationAssistant:
    def __init__(self):
        self.instruction_history = deque(maxlen=5)  # Store last 5 instructions
        self.last_instruction_time = 0
        self.instruction_delay = 3.0  # Minimum seconds between instructions
        self.current_instruction = ""
        self.priority_objects = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']
        
    def generate_instruction(self, objects_in_zones, path_status=None):
        """Generate navigation instruction based on detected objects and path status"""
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
        
        # Priority 3: No obstacles detected - use path status
        else:
            if path_status:
                if path_status == "Clear":
                    instruction = "Go straight"
                elif path_status == "Partially Blocked":
                    instruction = "Proceed with caution"
                elif path_status == "Fully Blocked":
                    instruction = "Stop and find another way"
                else:
                    instruction = "Path clear"
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
    
    def generate_directional_instruction(self, path_status, objects_in_zones):
        """Generate specific directional instructions based on path status and zones"""
        current_time = time.time()
        
        # Check if enough time has passed since last instruction
        if current_time - self.last_instruction_time < self.instruction_delay:
            return self.current_instruction
        
        instruction = ""
        
        if path_status == "Clear":
            instruction = "Go straight"
            
        elif path_status == "Partially Blocked":
            # Check which zones have obstacles
            left_blocked = len(objects_in_zones.get('left', [])) > 0
            right_blocked = len(objects_in_zones.get('right', [])) > 0
            center_blocked = len(objects_in_zones.get('center', [])) > 0
            
            if center_blocked:
                instruction = "Stop, obstacle ahead"
            elif left_blocked and not right_blocked:
                instruction = "Go right"
            elif right_blocked and not left_blocked:
                instruction = "Go left"
            elif left_blocked and right_blocked:
                instruction = "Find another way"
            else:
                instruction = "Proceed with caution"
                
        elif path_status == "Fully Blocked":
            # Check which zones are clear
            left_blocked = len(objects_in_zones.get('left', [])) > 0
            right_blocked = len(objects_in_zones.get('right', [])) > 0
            
            if not left_blocked:
                instruction = "Go left"
            elif not right_blocked:
                instruction = "Go right"
            else:
                instruction = "Stop and turn around"
        
        # Update instruction tracking
        if instruction != self.current_instruction:
            self.instruction_history.append(instruction)
            self.current_instruction = instruction
            self.last_instruction_time = current_time
        
        return instruction
    
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
    
    def generate_directional_instruction(self, path_status, zone_analysis):
        """Generate directional navigation instruction based on path status and zone analysis"""
        instruction = ""
        
        # Generate instruction based on path status and zone analysis
        if path_status == "Clear":
            instruction = "Go straight"
            
        elif path_status == "Left Blocked":
            # Direct instruction for left blocked
            instruction = "Go right"
            
        elif path_status == "Right Blocked":
            # Direct instruction for right blocked
            instruction = "Go left"
            
        elif path_status == "Partially Blocked":
            # Fallback for backward compatibility - use zone analysis
            left_blocked = zone_analysis.get('left', {}).get('blocked', False)
            center_blocked = zone_analysis.get('center', {}).get('blocked', False)
            right_blocked = zone_analysis.get('right', {}).get('blocked', False)
            
            if center_blocked:
                if left_blocked and not right_blocked:
                    instruction = "Go right"
                elif right_blocked and not left_blocked:
                    instruction = "Go left"
                elif left_blocked and right_blocked:
                    instruction = "Stop and find another way"
                else:
                    instruction = "Proceed with caution"
            else:
                if left_blocked and right_blocked:
                    instruction = "Stop and find another way"
                elif left_blocked:
                    instruction = "Go right"
                elif right_blocked:
                    instruction = "Go left"
                else:
                    instruction = "Go straight"
                    
        else:  # Fully Blocked
            instruction = "Stop and turn around"
        
        # Update instruction history and timing (only if instruction is different)
        if instruction != self.current_instruction:
            current_time = time.time()
            self.instruction_history.append(instruction)
            self.current_instruction = instruction
            self.last_instruction_time = current_time
        
        return instruction
