"""
Video Analysis GUI
Separate window for video file analysis
"""

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFrame, QSlider,
                            QMessageBox, QGridLayout, QGroupBox, QSizePolicy,
                            QProgressBar, QTextEdit, QFileDialog, QSpinBox,
                            QCheckBox, QTabWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import time
import os
from .video_analyzer import VideoAnalyzer
from config import SAFETY_COLORS

class VideoAnalysisWorker(QThread):
    """Worker thread for video analysis"""
    progress_updated = pyqtSignal(float, object)  # progress, frame_data
    analysis_complete = pyqtSignal(object)  # summary
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.voice_enabled = True
    
    def run(self):
        try:
            self.analyzer.process_video(
                progress_callback=self.progress_updated.emit,
                voice_enabled=self.voice_enabled
            )
            
            summary = self.analyzer.generate_summary()
            self.analysis_complete.emit(summary)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def set_voice_enabled(self, enabled):
        self.voice_enabled = enabled

class VideoAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Path Analysis")
        self.setGeometry(200, 200, 1000, 700)
        
        self.analyzer = VideoAnalyzer()
        self.object_detector = RealObjectDetector()
        self.worker = None
        self.current_video_path = None
        self.is_analyzing = False
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Tab 1: Video Analysis
        self.create_analysis_tab()
        
        # Tab 2: Results
        self.create_results_tab()
        
        # Tab 3: Export
        self.create_export_tab()
        
    def create_analysis_tab(self):
        """Create video analysis tab"""
        analysis_widget = QWidget()
        layout = QVBoxLayout(analysis_widget)
        
        # File selection group
        file_group = QGroupBox("Video File")
        file_layout = QHBoxLayout(file_group)
        
        self.file_label = QLabel("No video selected")
        self.file_label.setStyleSheet("font-weight: bold;")
        file_layout.addWidget(self.file_label)
        
        self.browse_btn = QPushButton("Browse Video")
        self.browse_btn.clicked.connect(self.browse_video)
        file_layout.addWidget(self.browse_btn)
        
        self.analyze_btn = QPushButton("Analyze Video")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        file_layout.addWidget(self.analyze_btn)
        
        layout.addWidget(file_group)
        
        # Video preview group
        preview_group = QGroupBox("Video Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("No video loaded")
        preview_layout.addWidget(self.video_label)
        
        layout.addWidget(preview_group)
        
        # Progress group
        progress_group = QGroupBox("Analysis Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)
        
        # Current frame info
        self.frame_info_label = QLabel("Frame: 0/0 | Time: 0.00s")
        progress_layout.addWidget(self.frame_info_label)
        
        # Current analysis result
        self.result_label = QLabel("Status: -- | Instruction: -- | Confidence: --")
        self.result_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        progress_layout.addWidget(self.result_label)
        
        layout.addWidget(progress_group)
        
        # Controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.voice_checkbox = QCheckBox("Enable Voice")
        self.voice_checkbox.setChecked(True)
        controls_layout.addWidget(self.voice_checkbox)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_analysis)
        self.pause_btn.setEnabled(False)
        controls_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)
        
        controls_layout.addStretch()
        
        layout.addWidget(controls_group)
        
        self.tab_widget.addTab(analysis_widget, "Video Analysis")
        
    def create_results_tab(self):
        """Create results tab"""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        
        # Summary group
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(200)
        summary_layout.addWidget(self.summary_text)
        
        layout.addWidget(summary_group)
        
        # Frame navigation
        nav_group = QGroupBox("Frame Navigation")
        nav_layout = QHBoxLayout(nav_group)
        
        nav_layout.addWidget(QLabel("Frame:"))
        
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setRange(0, 0)
        self.frame_spinbox.valueChanged.connect(self.show_frame)
        nav_layout.addWidget(self.frame_spinbox)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.show_frame)
        nav_layout.addWidget(self.frame_slider)
        
        nav_layout.addWidget(QLabel("/"))
        self.total_frames_label = QLabel("0")
        nav_layout.addWidget(self.total_frames_label)
        
        layout.addWidget(nav_group)
        
        # Frame details
        details_group = QGroupBox("Frame Details")
        details_layout = QVBoxLayout(details_group)
        
        self.frame_details_text = QTextEdit()
        self.frame_details_text.setReadOnly(True)
        details_layout.addWidget(self.frame_details_text)
        
        layout.addWidget(details_group)
        
        self.tab_widget.addTab(results_widget, "Results")
        
    def create_export_tab(self):
        """Create export tab"""
        export_widget = QWidget()
        layout = QVBoxLayout(export_widget)
        
        # Export options
        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout(options_group)
        
        # CSV export
        csv_layout = QHBoxLayout()
        self.csv_btn = QPushButton("Export Results to CSV")
        self.csv_btn.clicked.connect(self.export_csv)
        self.csv_btn.setEnabled(False)
        csv_layout.addWidget(self.csv_btn)
        csv_layout.addStretch()
        options_layout.addLayout(csv_layout)
        
        # Video export
        video_layout = QHBoxLayout()
        self.video_btn = QPushButton("Export Video with Overlays")
        self.video_btn.clicked.connect(self.export_video)
        self.video_btn.setEnabled(False)
        video_layout.addWidget(self.video_btn)
        video_layout.addStretch()
        options_layout.addLayout(video_layout)
        
        layout.addWidget(options_group)
        
        # Export status
        status_group = QGroupBox("Export Status")
        status_layout = QVBoxLayout(status_group)
        
        self.export_status_label = QLabel("No exports performed")
        self.export_status_label.setStyleSheet("padding: 10px;")
        status_layout.addWidget(self.export_status_label)
        
        layout.addWidget(status_group)
        
        layout.addStretch()
        
        self.tab_widget.addTab(export_widget, "Export")
        
    def browse_video(self):
        """Browse for video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, file_path):
        """Load video file"""
        try:
            if self.analyzer.load_video(file_path):
                self.current_video_path = file_path
                self.file_label.setText(f"Video: {os.path.basename(file_path)}")
                self.analyze_btn.setEnabled(True)
                
                # Update frame navigation
                info = self.analyzer.get_video_info()
                self.frame_spinbox.setRange(0, info['total_frames'] - 1)
                self.frame_slider.setRange(0, info['total_frames'] - 1)
                self.total_frames_label.setText(str(info['total_frames']))
                
                # Show first frame
                self.show_frame(0)
                
                self.export_status_label.setText("Video loaded successfully")
                
            else:
                QMessageBox.critical(self, "Error", "Failed to load video file")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading video: {str(e)}")
    
    def show_frame(self, frame_number):
        """Show specific frame with analysis"""
        if not self.current_video_path:
            return
        
        try:
            # Load frame from video
            cap = cv2.VideoCapture(self.current_video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Analyze frame
                frame_data = self.analyzer.analyze_frame(frame)
                
                if frame_data:
                    # Draw zones and objects
                    frame = self.analyzer.zone_detector.draw_zones(frame)
                    frame = self.analyzer.object_detector.draw_objects(frame, frame_data.get('detected_objects', []))
                    
                    # Update frame details
                    detected_objects = frame_data.get('detected_objects', [])
                    objects_text = "\n".join([f"• {obj['class']} (conf: {obj['confidence']:.2f})" for obj in detected_objects[:5]])
                    
                    details = f"""Frame: {frame_data['frame_number']}
Time: {frame_data['timestamp']:.2f}s
Path Status: {frame_data['path_status']}
Confidence: {frame_data['confidence']:.3f}
Instruction: {frame_data['instruction']}

Detected Objects:
{objects_text if objects_text else "No objects detected"}

Probabilities:
  Clear: {frame_data['probabilities']['clear']:.3f}
  Partial: {frame_data['probabilities']['partial']:.3f}
  Full: {frame_data['probabilities']['full']:.3f}

Zone Status:
  Left: {'Objects' if frame_data['zone_status']['left'] else 'Clear'}
  Center: {'Objects' if frame_data['zone_status']['center'] else 'Clear'}
  Right: {'Objects' if frame_data['zone_status']['right'] else 'Clear'}"""
                    
                    self.frame_details_text.setText(details)
                    self.result_label.setText(f"Status: {frame_data['path_status']} | "
                                            f"Instruction: {frame_data['instruction']} | "
                                            f"Confidence: {frame_data['confidence']:.3f}")
                    
                    # Draw overlays on frame
                    color = SAFETY_COLORS.get(frame_data['path_status'], (128, 128, 128))
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
                    
                    overlay = frame.copy()
                    cv2.addWeighted(overlay[0:80, :], 0.7, frame[0:80, :], 0.3, 0, frame[0:80, :])
                    
                    cv2.putText(frame, f"STATUS: {frame_data['path_status'].upper()}", (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"INSTRUCTION: {frame_data['instruction'].upper()}", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Update video label
                    self.update_video_display(frame)
                
                # Update frame info
                self.frame_info_label.setText(f"Frame: {frame_number}/{self.analyzer.total_frames} | "
                                            f"Time: {frame_number/self.analyzer.fps:.2f}s")
        
        except Exception as e:
            print(f"Error showing frame {frame_number}: {e}")
    
    def update_video_display(self, frame):
        """Update video display"""
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit display
        h, w = frame_rgb.shape[:2]
        display_w = 640
        display_h = 360
        
        if w > display_w or h > display_h:
            scale = min(display_w / w, display_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Convert to QImage
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)
    
    def start_analysis(self):
        """Start video analysis"""
        if not self.current_video_path:
            QMessageBox.warning(self, "Warning", "Please select a video file first")
            return
        
        # Reset UI
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting analysis...")
        self.is_analyzing = True
        
        # Update button states
        self.analyze_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        # Create and start worker
        self.worker = VideoAnalysisWorker(self.analyzer)
        self.worker.set_voice_enabled(self.voice_checkbox.isChecked())
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.analysis_complete.connect(self.on_analysis_complete)
        self.worker.error_occurred.connect(self.on_error_occurred)
        self.worker.start()
    
    def on_progress_updated(self, progress, frame_data):
        """Handle progress updates"""
        self.progress_bar.setValue(int(progress))
        self.progress_label.setText(f"Analyzing... {progress:.1f}%")
        
        if frame_data:
            self.frame_spinbox.setValue(frame_data['frame_number'])
            self.result_label.setText(f"Status: {frame_data['path_status']} | "
                                    f"Instruction: {frame_data['instruction']} | "
                                    f"Confidence: {frame_data['confidence']:.3f}")
    
    def on_analysis_complete(self, summary):
        """Handle analysis completion"""
        self.is_analyzing = False
        
        # Update button states
        self.analyze_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.csv_btn.setEnabled(True)
        self.video_btn.setEnabled(True)
        
        # Update summary
        summary_text = f"""Analysis Complete!
        
Total frames analyzed: {summary['total_frames']}
Video duration: {summary['duration']:.2f} seconds

Path Status Distribution:
"""
        for status, count in summary['status_distribution'].items():
            percentage = (count / summary['total_frames']) * 100
            summary_text += f"  {status}: {count} frames ({percentage:.1f}%)\n"
        
        summary_text += "\nNavigation Instructions:\n"
        for instruction, count in summary['instruction_distribution'].items():
            percentage = (count / summary['total_frames']) * 100
            summary_text += f"  {instruction}: {count} times ({percentage:.1f}%)\n"
        
        summary_text += f"\nMost common status: {summary['most_common_status']}"
        summary_text += f"\nMost common instruction: {summary['most_common_instruction']}"
        
        self.summary_text.setText(summary_text)
        self.progress_label.setText("Analysis complete!")
        
        QMessageBox.information(self, "Complete", "Video analysis completed successfully!")
    
    def on_error_occurred(self, error_message):
        """Handle analysis errors"""
        self.is_analyzing = False
        
        # Update button states
        self.analyze_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        
        self.progress_label.setText(f"Error: {error_message}")
        QMessageBox.critical(self, "Error", f"Analysis failed: {error_message}")
    
    def pause_analysis(self):
        """Pause/resume analysis"""
        if self.worker and self.worker.isRunning():
            if self.pause_btn.text() == "Pause":
                self.analyzer.pause()
                self.pause_btn.setText("Resume")
            else:
                self.analyzer.resume()
                self.pause_btn.setText("Pause")
    
    def stop_analysis(self):
        """Stop analysis"""
        if self.worker and self.worker.isRunning():
            self.analyzer.stop()
            self.worker.wait()
        
        self.is_analyzing = False
        
        # Update button states
        self.analyze_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        
        self.progress_label.setText("Analysis stopped")
    
    def export_csv(self):
        """Export results to CSV"""
        if not self.current_video_path:
            QMessageBox.warning(self, "Warning", "No video analyzed")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            if self.analyzer.export_results(file_path):
                self.export_status_label.setText(f"Results exported to: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Success", "Results exported successfully!")
            else:
                QMessageBox.critical(self, "Error", "Failed to export results")
    
    def export_video(self):
        """Export video with overlays"""
        if not self.current_video_path:
            QMessageBox.warning(self, "Warning", "No video analyzed")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "", "MP4 Files (*.mp4);;All Files (*)"
        )
        
        if file_path:
            self.export_status_label.setText("Creating video with overlays...")
            
            # Create video in separate thread to avoid blocking UI
            from PyQt5.QtCore import QThread
            
            class VideoExportWorker(QThread):
                finished = pyqtSignal(bool)
                
                def __init__(self, analyzer, output_path):
                    super().__init__()
                    self.analyzer = analyzer
                    self.output_path = output_path
                
                def run(self):
                    success = self.analyzer.create_video_with_overlays(self.output_path)
                    self.finished.emit(success)
            
            self.export_worker = VideoExportWorker(self.analyzer, file_path)
            self.export_worker.finished.connect(self.on_video_export_complete)
            self.export_worker.start()
    
    def on_video_export_complete(self, success):
        """Handle video export completion"""
        if success:
            self.export_status_label.setText("Video export completed successfully!")
            QMessageBox.information(self, "Success", "Video exported successfully!")
        else:
            self.export_status_label.setText("Video export failed")
            QMessageBox.critical(self, "Error", "Failed to export video")
    
    def closeEvent(self, event):
        """Handle window closing"""
        if self.is_analyzing:
            reply = QMessageBox.question(self, 'Quit', 'Analysis is in progress. Quit anyway?',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        # Cleanup
        if self.worker and self.worker.isRunning():
            self.analyzer.stop()
            self.worker.wait()
        
        self.analyzer.release()
        event.accept()
