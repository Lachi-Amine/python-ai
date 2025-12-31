import cv2
import numpy as np
import tensorflow as tf
import time
import os
import sys

try:
    import pyttsx3

    AUDIO_AVAILABLE = True
except ImportError:
    print("Warning: pyttsx3 not found. Audio disabled.")
    AUDIO_AVAILABLE = False


class BlindPathSystem:
    def __init__(self, model_filename='best_model.h5'):
        print("Initializing System...")

        if not os.path.exists(model_filename):
            print(f"Error: {model_filename} not found.")
            sys.exit(1)

        print(f"Loading model: {model_filename}")
        self.model = tf.keras.models.load_model(model_filename)

        self.engine = None
        if AUDIO_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
            except:
                pass

        self.last_speak_time = 0
        self.mode = "conservative"
        self.thresholds = {
            "conservative": 0.50,
            "balanced": 0.65,
            "aggressive": 0.80
        }
        self.start_time = time.time()
        self.frame_count = 0

    def speak(self, text):
        if not self.engine:
            return

        current_time = time.time()
        if current_time - self.last_speak_time > 2.5:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
                self.last_speak_time = current_time
            except:
                pass

    def get_fps(self):
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0

    def switch_mode(self):
        modes = list(self.thresholds.keys())
        idx = modes.index(self.mode)
        self.mode = modes[(idx + 1) % len(modes)]
        print(f"Mode switched to: {self.mode}")

    def process(self, frame):
        self.frame_count += 1

        resized = cv2.resize(frame, (224, 224))
        normalized = resized.astype('float32') / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)

        probs = self.model.predict(input_tensor, verbose=0)[0]
        p_clear, p_partial, p_block = probs

        thresh = self.thresholds[self.mode]

        status = "GREEN"
        message = "Path Clear"
        color = (0, 255, 0)

        if p_block > thresh:
            status = "RED"
            message = "STOP! BLOCKED"
            color = (0, 0, 255)
            self.speak("Stop. Way blocked.")
        elif p_partial > thresh:
            status = "YELLOW"
            message = "WARNING: Obstacle"
            color = (0, 255, 255)
            self.speak("Warning.")

        return status, message, color, probs


def main():
    system = BlindPathSystem()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    print("System Started. Press 'q' to Quit, 'm' to Switch Mode.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        status, message, color, probs = system.process(frame)

        display = frame.copy()

        cv2.rectangle(display, (0, 0), (640, 100), (0, 0, 0), -1)
        cv2.addWeighted(display[0:100, :], 0.8, frame[0:100, :], 0.2, 0, display[0:100, :])

        cv2.putText(display, f"STATUS: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display, message, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        stats = f"FPS: {system.get_fps():.1f} | Mode: {system.mode} | C:{probs[0]:.2f} P:{probs[1]:.2f} B:{probs[2]:.2f}"
        cv2.putText(display, stats, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Blind Path System', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            system.switch_mode()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()