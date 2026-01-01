"""
Path Detection System using MobileNet Model
Combines the academic project's MobileNet approach with simplified interface
"""

import cv2
import numpy as np
import tensorflow as tf
from config import MODEL_PATH, MODEL_INPUT_SHAPE, MODEL_CLASSES, CONFIDENCE_THRESHOLD, SAFETY_MODES, DEFAULT_SAFETY_MODE

class PathDetector:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or MODEL_PATH
        self.safety_mode = DEFAULT_SAFETY_MODE
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self):
        """Load the MobileNet model"""
        try:
            if not self.model_path.exists():
                print(f"Error: Model file not found at {self.model_path}")
                return False
            
            print(f"Loading model: {self.model_path}")
            
            # Custom object for DepthwiseConv2D compatibility
            def custom_depthwise_conv2d(*args, **kwargs):
                # Remove unsupported 'groups' argument
                if 'groups' in kwargs:
                    kwargs.pop('groups')
                return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)
            
            custom_objects = {
                'DepthwiseConv2D': custom_depthwise_conv2d
            }
            
            self.model = tf.keras.models.load_model(str(self.model_path), custom_objects=custom_objects)
            print("Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def set_safety_mode(self, mode):
        """Set the safety mode"""
        if mode in SAFETY_MODES:
            self.safety_mode = mode
            print(f"Safety mode set to: {mode}")
        else:
            print(f"Invalid safety mode: {mode}")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Resize to model input size
        resized = cv2.resize(frame, MODEL_INPUT_SHAPE[:2])
        # Normalize to [0, 1]
        normalized = resized.astype('float32') / 255.0
        # Add batch dimension
        input_tensor = np.expand_dims(normalized, axis=0)
        return input_tensor
    
    def predict_path_status(self, frame):
        """Predict path status from frame"""
        if self.model is None:
            return None, None, None
        
        try:
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            
            # Get prediction
            predictions = self.model.predict(input_tensor, verbose=0)[0]
            
            # Get class probabilities
            prob_clear = predictions[0]
            prob_partial = predictions[1] 
            prob_full = predictions[2]
            
            # Get current thresholds
            thresholds = SAFETY_MODES[self.safety_mode]
            
            # Determine safety status
            status = "Clear"
            confidence = prob_clear
            
            if prob_full > thresholds["full"]:
                status = "Fully Blocked"
                confidence = prob_full
            elif prob_partial > thresholds["partial"]:
                status = "Partially Blocked" 
                confidence = prob_partial
            
            # Create result dictionary
            result = {
                'status': status,
                'confidence': float(confidence),
                'probabilities': {
                    'clear': float(prob_clear),
                    'partial': float(prob_partial),
                    'full': float(prob_full)
                },
                'safety_mode': self.safety_mode
            }
            
            return result, predictions, status
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None, None
    
    def get_safety_color(self, status):
        """Get color for safety status"""
        colors = {
            "Clear": (0, 255, 0),           # Green
            "Partially Blocked": (0, 255, 255),  # Yellow
            "Fully Blocked": (0, 0, 255)   # Red
        }
        return colors.get(status, (128, 128, 128))  # Gray default
    
    def is_model_ready(self):
        """Check if model is loaded and ready"""
        return self.model is not None
