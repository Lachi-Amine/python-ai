import tkinter as tk
import sys
import traceback

print("Step 1: Basic imports OK")

try:
    from camera import CameraCapture
    print("Step 2: Camera import OK")
except Exception as e:
    print(f"Camera import error: {e}")
    sys.exit(1)

try:
    from simple_detector import SimpleObjectDetector as ObjectDetector
    print("Step 3: Detector import OK")
except Exception as e:
    print(f"Detector import error: {e}")
    sys.exit(1)

try:
    from zone_detector import ZoneDetector
    print("Step 4: Zone detector import OK")
except Exception as e:
    print(f"Zone detector import error: {e}")
    sys.exit(1)

try:
    from navigation import NavigationAssistant
    print("Step 5: Navigation import OK")
except Exception as e:
    print(f"Navigation import error: {e}")
    sys.exit(1)

try:
    from voice_guide import VoiceGuide
    print("Step 6: Voice guide import OK")
except Exception as e:
    print(f"Voice guide import error: {e}")
    sys.exit(1)

try:
    from gui import AssistiveGUI
    print("Step 7: GUI import OK")
except Exception as e:
    print(f"GUI import error: {e}")
    sys.exit(1)

print("All imports successful!")

try:
    print("Step 8: Creating Tkinter root")
    root = tk.Tk()
    print("Step 9: Tkinter root created")
    
    print("Step 10: Creating GUI")
    gui = AssistiveGUI(root)
    print("Step 11: GUI created")
    
    print("Step 12: Creating camera")
    camera = CameraCapture()
    print("Step 13: Camera created")
    
    print("Step 14: Creating detector")
    detector = ObjectDetector()
    print("Step 15: Detector created")
    
    print("Step 16: Creating zone detector")
    zone_detector = ZoneDetector()
    print("Step 17: Zone detector created")
    
    print("Step 18: Creating navigation")
    navigation = NavigationAssistant()
    print("Step 19: Navigation created")
    
    print("Step 20: Creating voice guide")
    voice_guide = VoiceGuide()
    print("Step 21: Voice guide created")
    
    print("All components created successfully!")
    
except Exception as e:
    print(f"Component creation error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Debug completed successfully!")
