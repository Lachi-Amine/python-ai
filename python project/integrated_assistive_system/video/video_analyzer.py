"""
Video File Analyzer Module
Analyzes video files with path detection and directional navigation
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
from core.path_detector import PathDetector
from core.zone_detector import ZoneDetector
from core.navigation import NavigationAssistant
from utils.voice_guide import VoiceGuide
from config import CAMERA_WIDTH, CAMERA_HEIGHT

class VideoAnalyzer:
    def __init__(self):
        self.detector = PathDetector()
        self.zone_detector = ZoneDetector()
        self.navigation = NavigationAssistant()
        self.voice_guide = VoiceGuide()
        
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30
        self.is_playing = False
        self.is_paused = False
        
        # Analysis results
        self.results = []
        self.frame_data = []
        
    def load_video(self, video_path):
        """Load video file for analysis"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                raise Exception(f"Could not open video file: {video_path}")
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video loaded: {video_path}")
            print(f"Resolution: {width}x{height}")
            print(f"Total frames: {self.total_frames}")
            print(f"FPS: {self.fps}")
            
            return True
            
        except Exception as e:
            print(f"Error loading video: {e}")
            return False
    
    def analyze_frame(self, frame):
        """Analyze a single frame"""
        if frame is None:
            return None
        
        # Update frame size for zone detector
        height, width = frame.shape[:2]
        self.zone_detector.update_frame_size(width, height)
        
        # Predict path status
        result, predictions, status = self.detector.predict_path_status(frame)
        
        if result:
            # Update frame size for zone detector
            height, width = frame.shape[:2]
            self.zone_detector.update_frame_size(width, height)
            
            # Detect real objects in the frame
            detected_objects = self.object_detector.detect_objects(frame)
            navigation_objects = self.object_detector.filter_navigation_objects(detected_objects)
            
            # Categorize objects into zones
            objects_in_zones = self.zone_detector.categorize_objects(navigation_objects)
            
            # Generate directional navigation instruction
            instruction = self.navigation.generate_directional_instruction(result['status'], objects_in_zones)
            
            # Create frame data
            frame_data = {
                'frame_number': self.current_frame,
                'timestamp': self.current_frame / self.fps,
                'path_status': result['status'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'instruction': instruction,
                'objects_in_zones': objects_in_zones,
                'zone_status': self.zone_detector.get_zone_status(),
                'detected_objects': navigation_objects
            }
            
            return frame_data
        
        return None
    
    def process_video(self, progress_callback=None, voice_enabled=True):
        """Process entire video and collect results"""
        if self.cap is None:
            raise Exception("No video loaded")
        
        print("Starting video analysis...")
        self.results = []
        self.frame_data = []
        self.current_frame = 0
        self.is_playing = True
        
        start_time = time.time()
        last_voice_time = 0
        
        try:
            while self.is_playing and self.current_frame < self.total_frames:
                # Set frame position
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Analyze frame
                frame_result = self.analyze_frame(frame)
                if frame_result:
                    self.frame_data.append(frame_result)
                    
                    # Voice guidance (every 3 seconds of video time)
                    if voice_enabled:
                        current_video_time = frame_result['timestamp']
                        if current_video_time - last_voice_time > 3.0:
                            self.voice_guide.speak(frame_result['instruction'])
                            last_voice_time = current_video_time
                
                # Progress callback
                if progress_callback:
                    progress = (self.current_frame / self.total_frames) * 100
                    progress_callback(progress, frame_result)
                
                self.current_frame += 1
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error during video processing: {e}")
            raise
        
        finally:
            self.is_playing = False
            processing_time = time.time() - start_time
            
            print(f"Video analysis complete!")
            print(f"Processed {len(self.frame_data)} frames")
            print(f"Processing time: {processing_time:.2f} seconds")
            
            # Generate summary statistics
            self.generate_summary()
    
    def generate_summary(self):
        """Generate summary statistics from analysis results"""
        if not self.frame_data:
            return
        
        # Count different path statuses
        status_counts = {}
        instruction_counts = {}
        
        for data in self.frame_data:
            status = data['path_status']
            instruction = data['instruction']
            
            status_counts[status] = status_counts.get(status, 0) + 1
            instruction_counts[instruction] = instruction_counts.get(instruction, 0) + 1
        
        # Calculate percentages
        total_frames = len(self.frame_data)
        
        print("\n=== Video Analysis Summary ===")
        print(f"Total frames analyzed: {total_frames}")
        print(f"Video duration: {self.total_frames / self.fps:.2f} seconds")
        
        print("\nPath Status Distribution:")
        for status, count in status_counts.items():
            percentage = (count / total_frames) * 100
            print(f"  {status}: {count} frames ({percentage:.1f}%)")
        
        print("\nNavigation Instructions:")
        for instruction, count in sorted(instruction_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_frames) * 100
            print(f"  {instruction}: {count} times ({percentage:.1f}%)")
        
        # Find most common status and instruction
        most_common_status = max(status_counts, key=status_counts.get)
        most_common_instruction = max(instruction_counts, key=instruction_counts.get)
        
        print(f"\nMost common path status: {most_common_status}")
        print(f"Most common instruction: {most_common_instruction}")
        
        return {
            'total_frames': total_frames,
            'duration': self.total_frames / self.fps,
            'status_distribution': status_counts,
            'instruction_distribution': instruction_counts,
            'most_common_status': most_common_status,
            'most_common_instruction': most_common_instruction
        }
    
    def get_frame_at_position(self, frame_number):
        """Get analyzed data for specific frame"""
        for data in self.frame_data:
            if data['frame_number'] == frame_number:
                return data
        return None
    
    def export_results(self, output_path):
        """Export analysis results to CSV file"""
        if not self.frame_data:
            print("No data to export")
            return False
        
        try:
            import csv
            
            with open(output_path, 'w', newline='') as csvfile:
                fieldnames = ['frame_number', 'timestamp', 'path_status', 'confidence', 
                             'instruction', 'clear_prob', 'partial_prob', 'full_prob']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for data in self.frame_data:
                    writer.writerow({
                        'frame_number': data['frame_number'],
                        'timestamp': f"{data['timestamp']:.2f}",
                        'path_status': data['path_status'],
                        'confidence': f"{data['confidence']:.3f}",
                        'instruction': data['instruction'],
                        'clear_prob': f"{data['probabilities']['clear']:.3f}",
                        'partial_prob': f"{data['probabilities']['partial']:.3f}",
                        'full_prob': f"{data['probabilities']['full']:.3f}"
                    })
            
            print(f"Results exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False
    
    def create_video_with_overlays(self, output_path):
        """Create output video with analysis overlays"""
        if self.cap is None or not self.frame_data:
            print("No video or data available")
            return False
        
        try:
            # Reset video capture
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            print(f"Creating output video: {output_path}")
            
            frame_data_dict = {data['frame_number']: data for data in self.frame_data}
            
            for frame_num in range(self.total_frames):
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Get analysis data for this frame
                frame_data = frame_data_dict.get(frame_num)
                
                if frame_data:
                    # Draw status overlay
                    status = frame_data['path_status']
                    confidence = frame_data['confidence']
                    instruction = frame_data['instruction']
                    
                    # Get color based on status
                    colors = {
                        "Clear": (0, 255, 0),
                        "Partially Blocked": (0, 255, 255),
                        "Fully Blocked": (0, 0, 255)
                    }
                    color = colors.get(status, (128, 128, 128))
                    
                    # Draw overlay
                    cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)
                    overlay = frame.copy()
                    cv2.addWeighted(overlay[0:100, :], 0.7, frame[0:100, :], 0.3, 0, frame[0:100, :])
                    
                    # Draw text
                    cv2.putText(frame, f"STATUS: {status.upper()}", (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"INSTRUCTION: {instruction.upper()}", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"Confidence: {confidence:.2f} | Frame: {frame_num}", 
                               (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    
                    # Draw zones
                    frame = self.zone_detector.draw_zones(frame)
                
                # Write frame
                out.write(frame)
            
            out.release()
            print(f"Output video saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating output video: {e}")
            return False
    
    def pause(self):
        """Pause video processing"""
        self.is_paused = True
    
    def resume(self):
        """Resume video processing"""
        self.is_paused = False
    
    def stop(self):
        """Stop video processing"""
        self.is_playing = False
    
    def release(self):
        """Release video resources"""
        if self.cap:
            self.cap.release()
        self.voice_guide.cleanup()
    
    def get_video_info(self):
        """Get video information"""
        if self.cap is None:
            return {}
        
        return {
            'path': self.video_path,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'duration': self.total_frames / self.fps if self.fps > 0 else 0,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
