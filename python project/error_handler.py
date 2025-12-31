import logging
import traceback
import sys
import os
from datetime import datetime

class ErrorHandler:
    """Centralized error handling and logging for the assistive system"""
    
    def __init__(self, log_file='assistive_system.log'):
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        try:
            # Create logs directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(self.log_file),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            
            self.logger = logging.getLogger(__name__)
            self.logger.info("Error handler initialized")
            
        except Exception as e:
            print(f"Failed to setup logging: {e}")
    
    def log_error(self, error, context=""):
        """Log an error with context"""
        error_msg = f"{context}: {str(error)}"
        if hasattr(error, '__traceback__'):
            error_msg += f"\n{traceback.format_exc()}"
        
        self.logger.error(error_msg)
    
    def log_warning(self, message, context=""):
        """Log a warning message"""
        warning_msg = f"{context}: {message}"
        self.logger.warning(warning_msg)
    
    def log_info(self, message, context=""):
        """Log an info message"""
        info_msg = f"{context}: {message}"
        self.logger.info(info_msg)
    
    def handle_camera_error(self, error):
        """Handle camera-related errors"""
        self.log_error(error, "Camera Error")
        return {
            'type': 'camera',
            'message': 'Camera initialization failed. Please check camera connection.',
            'recoverable': True
        }
    
    def handle_model_error(self, error):
        """Handle model-related errors"""
        self.log_error(error, "Model Error")
        return {
            'type': 'model',
            'message': 'Failed to load YOLO model. Please check model file.',
            'recoverable': False
        }
    
    def handle_voice_error(self, error):
        """Handle voice-related errors"""
        self.log_error(error, "Voice Error")
        return {
            'type': 'voice',
            'message': 'Voice system error. Voice guidance disabled.',
            'recoverable': True
        }
    
    def handle_gui_error(self, error):
        """Handle GUI-related errors"""
        self.log_error(error, "GUI Error")
        return {
            'type': 'gui',
            'message': 'GUI error occurred. Some features may not work.',
            'recoverable': True
        }
    
    def handle_processing_error(self, error):
        """Handle processing loop errors"""
        self.log_error(error, "Processing Error")
        return {
            'type': 'processing',
            'message': 'Processing error occurred. System will retry.',
            'recoverable': True
        }
    
    def handle_critical_error(self, error):
        """Handle critical system errors"""
        self.log_error(error, "Critical Error")
        return {
            'type': 'critical',
            'message': 'Critical system error. Application will exit.',
            'recoverable': False
        }
    
    def get_error_summary(self):
        """Get a summary of recent errors from log file"""
        try:
            if not os.path.exists(self.log_file):
                return "No error log found"
            
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Get last 20 error lines
            error_lines = [line for line in lines[-20:] if 'ERROR' in line]
            
            if error_lines:
                return f"Recent errors ({len(error_lines)}):\n" + '\n'.join(error_lines[-5:])
            else:
                return "No recent errors"
                
        except Exception as e:
            return f"Error reading log file: {e}"
    
    def clear_logs(self):
        """Clear the log file"""
        try:
            if os.path.exists(self.log_file):
                open(self.log_file, 'w').close()
                self.log_info("Log file cleared")
                return True
            return False
        except Exception as e:
            self.log_error(e, "Log Clear Error")
            return False

# Global error handler instance
error_handler = ErrorHandler()

def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    error_handler.log_error(exc_value, "Unhandled Exception")
    error_handler.log_error(''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)), "Traceback")

# Set global exception handler
sys.excepthook = handle_exception
