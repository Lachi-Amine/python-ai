import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time

class AssistiveGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Assistive Navigation System for Blind Users")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')
        
        # GUI variables
        self.camera_active = False
        self.detection_active = False
        self.voice_active = True
        self.current_frame = None
        self.processed_frame = None
        
        # Status variables
        self.status_text = tk.StringVar(value="System Ready")
        self.instruction_text = tk.StringVar(value="No instruction")
        self.object_count_text = tk.StringVar(value="Objects: 0")
        
        # Create GUI components
        self.create_widgets()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        """Create all GUI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Camera feed frame
        camera_frame = ttk.LabelFrame(main_frame, text="Camera View", padding="5")
        camera_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.camera_label = tk.Label(camera_frame, bg='black', width=640, height=480)
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # System controls
        ttk.Label(control_frame, text="System Controls", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        self.start_camera_btn = ttk.Button(control_frame, text="Start Camera", command=self.toggle_camera)
        self.start_camera_btn.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.start_detection_btn = ttk.Button(control_frame, text="Start Detection", command=self.toggle_detection)
        self.start_detection_btn.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.voice_btn = ttk.Button(control_frame, text="Voice: ON", command=self.toggle_voice)
        self.voice_btn.grid(row=3, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Status display
        ttk.Separator(control_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(control_frame, text="Status", font=('Arial', 12, 'bold')).grid(row=5, column=0, columnspan=2, pady=5)
        
        status_label = ttk.Label(control_frame, textvariable=self.status_text, foreground='green')
        status_label.grid(row=6, column=0, columnspan=2, pady=5)
        
        # Instruction display
        ttk.Separator(control_frame, orient='horizontal').grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(control_frame, text="Navigation Instruction", font=('Arial', 12, 'bold')).grid(row=8, column=0, columnspan=2, pady=5)
        
        instruction_label = ttk.Label(control_frame, textvariable=self.instruction_text, 
                                    font=('Arial', 14, 'bold'), foreground='blue', wraplength=200)
        instruction_label.grid(row=9, column=0, columnspan=2, pady=5)
        
        # Object count
        ttk.Separator(control_frame, orient='horizontal').grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(control_frame, text="Detection Info", font=('Arial', 12, 'bold')).grid(row=11, column=0, columnspan=2, pady=5)
        
        object_count_label = ttk.Label(control_frame, textvariable=self.object_count_text)
        object_count_label.grid(row=12, column=0, columnspan=2, pady=5)
        
        # Zone indicators
        self.zone_frame = ttk.LabelFrame(control_frame, text="Zone Status", padding="5")
        self.zone_frame.grid(row=13, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.zone_labels = {}
        zones = ['Left', 'Center', 'Right']
        for i, zone in enumerate(zones):
            label = ttk.Label(self.zone_frame, text=f"{zone}: Clear", foreground='green')
            label.grid(row=i, column=0, sticky=tk.W, pady=2)
            self.zone_labels[zone.lower()] = label
        
        # Settings
        ttk.Separator(control_frame, orient='horizontal').grid(row=14, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(control_frame, text="Settings", font=('Arial', 12, 'bold')).grid(row=15, column=0, columnspan=2, pady=5)
        
        # Confidence threshold
        ttk.Label(control_frame, text="Confidence:").grid(row=16, column=0, sticky=tk.W, pady=2)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(control_frame, from_=0.1, to=1.0, variable=self.confidence_var, 
                                   orient=tk.HORIZONTAL, length=150)
        confidence_scale.grid(row=16, column=1, pady=2)
        
        # Voice rate
        ttk.Label(control_frame, text="Voice Speed:").grid(row=17, column=0, sticky=tk.W, pady=2)
        self.voice_rate_var = tk.IntVar(value=150)
        voice_scale = ttk.Scale(control_frame, from_=50, to=300, variable=self.voice_rate_var, 
                              orient=tk.HORIZONTAL, length=150)
        voice_scale.grid(row=17, column=1, pady=2)
        
        # Test buttons
        ttk.Separator(control_frame, orient='horizontal').grid(row=18, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        test_voice_btn = ttk.Button(control_frame, text="Test Voice", command=self.test_voice)
        test_voice_btn.grid(row=19, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
    def update_camera_feed(self, frame):
        """Update the camera feed display"""
        if frame is None:
            return
            
        try:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit display
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            
            # Convert to PIL Image
            image = Image.fromarray(frame_resized)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=image)
            
            # Update label
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error updating camera feed: {e}")
    
    def update_status(self, status, color='green'):
        """Update status text"""
        self.status_text.set(status)
        # You could add color logic here if needed
    
    def update_instruction(self, instruction):
        """Update navigation instruction"""
        self.instruction_text.set(instruction)
    
    def update_object_count(self, count):
        """Update object count display"""
        self.object_count_text.set(f"Objects: {count}")
    
    def update_zone_status(self, zone_status):
        """Update zone status indicators"""
        for zone, has_objects in zone_status.items():
            if zone in self.zone_labels:
                if has_objects:
                    self.zone_labels[zone].configure(text=f"{zone.title()}: Object", foreground='red')
                else:
                    self.zone_labels[zone].configure(text=f"{zone.title()}: Clear", foreground='green')
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        self.camera_active = not self.camera_active
        if self.camera_active:
            self.start_camera_btn.configure(text="Stop Camera")
            self.update_status("Camera Active")
        else:
            self.start_camera_btn.configure(text="Start Camera")
            self.update_status("Camera Stopped")
            # Clear camera feed
            self.camera_label.configure(image='', bg='black')
    
    def toggle_detection(self):
        """Toggle detection on/off"""
        self.detection_active = not self.detection_active
        if self.detection_active:
            self.start_detection_btn.configure(text="Stop Detection")
            self.update_status("Detection Active")
        else:
            self.start_detection_btn.configure(text="Start Detection")
            self.update_status("Detection Stopped")
    
    def toggle_voice(self):
        """Toggle voice guidance on/off"""
        self.voice_active = not self.voice_active
        if self.voice_active:
            self.voice_btn.configure(text="Voice: ON")
            self.update_status("Voice Enabled")
        else:
            self.voice_btn.configure(text="Voice: OFF")
            self.update_status("Voice Disabled")
    
    def test_voice(self):
        """Test voice functionality"""
        self.update_status("Testing Voice...")
        # This will be handled by the main application
    
    def get_confidence_threshold(self):
        """Get confidence threshold value"""
        return self.confidence_var.get()
    
    def get_voice_rate(self):
        """Get voice rate value"""
        return self.voice_rate_var.get()
    
    def show_error(self, title, message):
        """Show error message"""
        messagebox.showerror(title, message)
    
    def show_info(self, title, message):
        """Show info message"""
        messagebox.showinfo(title, message)
    
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit the assistive system?"):
            self.camera_active = False
            self.detection_active = False
            self.voice_active = False
            self.root.destroy()
    
    def reset_display(self):
        """Reset display to initial state"""
        self.camera_label.configure(image='', bg='black')
        self.update_status("System Ready")
        self.update_instruction("No instruction")
        self.update_object_count(0)
        
        # Reset zone status
        for zone in self.zone_labels:
            self.zone_labels[zone].configure(text=f"{zone.title()}: Clear", foreground='green')
