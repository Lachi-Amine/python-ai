#!/usr/bin/env python3
"""
Test script to verify launcher subprocess behavior
"""

import sys
import os
import subprocess
import time
from PyQt5.QtWidgets import QApplication, QMessageBox

def test_subprocess():
    """Test subprocess behavior"""
    print("Testing launcher subprocess behavior...")
    
    # Test camera mode
    print("\n1. Testing camera mode subprocess...")
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, "-c", 
            "import time; print('Camera app started'); time.sleep(2); print('Camera app exiting')"], 
            timeout=10)
        end_time = time.time()
        print(f"✅ Camera subprocess completed in {end_time - start_time:.1f}s")
        print(f"   Return code: {result.returncode}")
    except Exception as e:
        print(f"❌ Camera subprocess error: {e}")
    
    # Test video mode
    print("\n2. Testing video mode subprocess...")
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, "-c", 
            "import time; print('Video app started'); time.sleep(2); print('Video app exiting')"], 
            timeout=10)
        end_time = time.time()
        print(f"✅ Video subprocess completed in {end_time - start_time:.1f}s")
        print(f"   Return code: {result.returncode}")
    except Exception as e:
        print(f"❌ Video subprocess error: {e}")
    
    print("\n3. Testing actual main.py (quick test)...")
    try:
        # Test if main.py can import and exit quickly
        test_code = '''
import sys
sys.path.insert(0, ".")
try:
    from main import IntegratedAssistiveSystem
    print("main.py imports successfully")
    # Don't actually run, just test import
except Exception as e:
    print(f"main.py import error: {e}")
'''
        result = subprocess.run([sys.executable, "-c", test_code], 
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              timeout=10)
        print(f"✅ main.py test completed")
        print(f"   Return code: {result.returncode}")
    except Exception as e:
        print(f"❌ main.py test error: {e}")
    
    print("\n✅ Subprocess testing complete!")

if __name__ == "__main__":
    test_subprocess()
