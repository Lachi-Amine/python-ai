"""
Voice Guide Module
Simplified text-to-speech for navigation instructions
"""

import pyttsx3
import threading
import queue
import time
from config import VOICE_ENABLED, VOICE_RATE, VOICE_VOLUME, INSTRUCTION_DELAY, VOICE_MESSAGES

class VoiceGuide:
    def __init__(self):
        self.engine = None
        self.voice_queue = queue.Queue()
        self.is_speaking = False
        self.enabled = VOICE_ENABLED
        self.last_spoken_text = ""
        self.speech_rate = VOICE_RATE
        self.volume = VOICE_VOLUME
        self.last_instruction_time = 0
        
        # Initialize on creation
        self.initialize()
    
    def initialize(self):
        """Initialize the text-to-speech engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Configure voice properties
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find a female voice (often clearer for navigation)
                for voice in voices:
                    if 'female' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice if no female voice found
                    self.engine.setProperty('voice', voices[0].id)
            
            # Set speech rate and volume
            self.engine.setProperty('rate', self.speech_rate)
            self.engine.setProperty('volume', self.volume)
            
            # Start speech thread
            speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
            speech_thread.start()
            
            print("Voice guide initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing voice guide: {e}")
            self.enabled = False
            return False
    
    def _speech_worker(self):
        """Background thread for handling speech synthesis"""
        while True:
            try:
                # Get text from queue
                text = self.voice_queue.get(timeout=0.1)
                
                if text and self.enabled and self.engine:
                    self.is_speaking = True
                    self.last_spoken_text = text
                    
                    # Speak the text
                    self.engine.say(text)
                    self.engine.runAndWait()
                    
                    self.is_speaking = False
                
                self.voice_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Speech error: {e}")
                self.is_speaking = False
    
    def speak(self, text, priority=False):
        """Add text to speech queue"""
        if not text or not self.enabled:
            return
        
        # Avoid repeating the same instruction
        if text == self.last_spoken_text and not priority:
            return
        
        # Check instruction delay
        current_time = time.time()
        if not priority and (current_time - self.last_instruction_time < INSTRUCTION_DELAY):
            return
        
        try:
            if priority:
                # Clear queue and add priority message
                while not self.voice_queue.empty():
                    try:
                        self.voice_queue.get_nowait()
                        self.voice_queue.task_done()
                    except queue.Empty:
                        break
            
            self.voice_queue.put(text)
            self.last_instruction_time = current_time
            
        except Exception as e:
            print(f"Error adding text to speech queue: {e}")
    
    def speak_path_status(self, status, is_emergency=False):
        """Speak path status with appropriate message"""
        if status in VOICE_MESSAGES:
            message = VOICE_MESSAGES[status]
            if is_emergency and status == "Fully Blocked":
                message = "Warning! " + message
            self.speak(message, priority=is_emergency)
    
    def stop_speaking(self):
        """Stop current speech and clear queue"""
        try:
            if self.engine:
                self.engine.stop()
            
            # Clear the queue
            while not self.voice_queue.empty():
                try:
                    self.voice_queue.get_nowait()
                    self.voice_queue.task_done()
                except queue.Empty:
                    break
                    
            self.is_speaking = False
            
        except Exception as e:
            print(f"Error stopping speech: {e}")
    
    def set_voice_rate(self, rate):
        """Set speech rate (words per minute)"""
        if 50 <= rate <= 300:
            self.speech_rate = rate
            if self.engine:
                self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume):
        """Set speech volume (0.0 to 1.0)"""
        if 0.0 <= volume <= 1.0:
            self.volume = volume
            if self.engine:
                self.engine.setProperty('volume', volume)
    
    def enable(self):
        """Enable voice guidance"""
        self.enabled = True
    
    def disable(self):
        """Disable voice guidance"""
        self.stop_speaking()
        self.enabled = False
    
    def is_enabled(self):
        """Check if voice guidance is enabled"""
        return self.enabled
    
    def is_currently_speaking(self):
        """Check if currently speaking"""
        return self.is_speaking
    
    def cleanup(self):
        """Cleanup voice guide resources"""
        self.stop_speaking()
        self.enabled = False
        if self.engine:
            self.engine.stop()
