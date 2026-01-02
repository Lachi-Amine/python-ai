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
            
            # Handle .keras file (actually H5 format)
            model_path = str(self.model_path)
            if self.model_path.suffix == '.keras':
                # Copy to temporary H5 file to load properly
                import tempfile
                import shutil
                with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                    shutil.copy2(model_path, tmp_file.name)
                    temp_path = tmp_file.name
                
                try:
                    model = self._load_model_with_custom_objects(temp_path)
                    print("Model loaded successfully from .keras file")
                    self.model = model
                    return True
                finally:
                    import os
                    os.unlink(temp_path)
            
            # Load normally for other formats
            self.model = self._load_model_with_custom_objects(model_path)
            print("Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _load_model_with_custom_objects(self, model_path):
        """Load model with custom objects for compatibility"""
        # Custom object for DepthwiseConv2D compatibility
        def custom_depthwise_conv2d(*args, **kwargs):
            # Remove unsupported 'groups' argument
            if 'groups' in kwargs:
                kwargs.pop('groups')
            return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)
        
        custom_objects = {
            'DepthwiseConv2D': custom_depthwise_conv2d
        }
        
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
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
            
            # Add timeout protection for prediction
            import signal
            import threading
            
            prediction_result = [None, None]
            prediction_error = [None]
            
            def predict_thread():
                try:
                    predictions = self.model.predict(input_tensor, verbose=0)[0]
                    prediction_result[0] = predictions
                except Exception as e:
                    prediction_error[0] = e
            
            # Start prediction thread
            thread = threading.Thread(target=predict_thread)
            thread.daemon = True
            thread.start()
            
            # Wait for prediction with timeout (2 seconds max)
            thread.join(timeout=2.0)
            
            if thread.is_alive():
                print("Prediction timeout - skipping this frame")
                return None, None, None
            
            if prediction_error[0]:
                raise prediction_error[0]
            
            if prediction_result[0] is None:
                print("Prediction failed - no result")
                return None, None, None
            
            predictions = prediction_result[0]
            
            # Get class probabilities for 4 classes
            prob_clear = predictions[0]
            prob_left_blocked = predictions[1]
            prob_right_blocked = predictions[2]
            prob_full = predictions[3]
            
            # Get current thresholds
            thresholds = SAFETY_MODES[self.safety_mode]
            
            # Determine safety status
            status = "Clear"
            confidence = prob_clear
            
            if prob_full > thresholds["full"]:
                status = "Fully Blocked"
                confidence = prob_full
            elif prob_left_blocked > thresholds["left_blocked"]:
                status = "Left Blocked"
                confidence = prob_left_blocked
            elif prob_right_blocked > thresholds["right_blocked"]:
                status = "Right Blocked"
                confidence = prob_right_blocked
            
            # Create result dictionary
            result = {
                'status': status,
                'confidence': float(confidence),
                'probabilities': {
                    'clear': float(prob_clear),
                    'left_blocked': float(prob_left_blocked),
                    'right_blocked': float(prob_right_blocked),
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
