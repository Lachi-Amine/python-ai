import os
import json

class Config:
    """Configuration settings for the assistive navigation system"""
    
    # Camera settings
    CAMERA_ID = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # YOLO model settings
    MODEL_PATH = 'yolov8n.pt'
    CONFIDENCE_THRESHOLD = 0.5
    NAVIGATION_CLASSES = [
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
    ]
    
    # Zone detection settings
    ZONE_DIVISION = 3  # Divide frame into 3 zones (left, center, right)
    MIN_OBJECT_AREA = 1000  # Minimum area for object to be considered
    CLOSE_OBJECT_THRESHOLD = 10000  # Area threshold for "close" objects
    VERY_CLOSE_OBJECT_THRESHOLD = 15000  # Area threshold for "very close" objects
    EMERGENCY_OBJECT_THRESHOLD = 20000  # Area threshold for emergency situations
    
    # Navigation settings
    INSTRUCTION_DELAY = 3.0  # Minimum seconds between instructions
    PRIORITY_OBJECTS = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']
    
    # Voice settings
    VOICE_ENABLED = True
    VOICE_RATE = 150  # Words per minute
    VOICE_VOLUME = 0.8
    PREFER_FEMALE_VOICE = True
    
    # GUI settings
    WINDOW_WIDTH = 900
    WINDOW_HEIGHT = 700
    CAMERA_DISPLAY_WIDTH = 640
    CAMERA_DISPLAY_HEIGHT = 480
    
    # System settings
    PROCESSING_FPS = 30  # Target FPS for processing loop
    ERROR_RETRY_DELAY = 1.0  # Seconds to wait after error before retrying
    
    @classmethod
    def load_from_file(cls, config_file='config.json'):
        """Load configuration from JSON file"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update class attributes with loaded data
                for key, value in config_data.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)
                
                print(f"Configuration loaded from {config_file}")
                return True
            else:
                print(f"Config file {config_file} not found, using defaults")
                return False
                
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    @classmethod
    def save_to_file(cls, config_file='config.json'):
        """Save current configuration to JSON file"""
        try:
            config_data = {}
            
            # Get all class attributes that are uppercase (constants)
            for attr in dir(cls):
                if not attr.startswith('_') and attr.isupper():
                    config_data[attr] = getattr(cls, attr)
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            print(f"Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    @classmethod
    def create_default_config(cls, config_file='config.json'):
        """Create a default configuration file"""
        return cls.save_to_file(config_file)
    
    @classmethod
    def get_config_summary(cls):
        """Get a summary of current configuration"""
        return {
            'camera': f"{cls.CAMERA_WIDTH}x{cls.CAMERA_HEIGHT} @ {cls.CAMERA_FPS}fps",
            'model': cls.MODEL_PATH,
            'confidence': cls.CONFIDIDENCE_THRESHOLD if hasattr(cls, 'CONFIDIDENCE_THRESHOLD') else cls.CONFIDENCE_THRESHOLD,
            'voice': f"Rate: {cls.VOICE_RATE}, Volume: {cls.VOICE_VOLUME}",
            'zones': f"{cls.ZONE_DIVISION} zones"
        }
