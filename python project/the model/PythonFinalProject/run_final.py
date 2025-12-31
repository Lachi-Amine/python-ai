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
    print("Warning: pyttsx3 not installed. Audio disabled.")
    AUDIO_AVAILABLE = False


class BlindPathSystem:
    def __init__(self, model_filename='best_model.h5'):
        print("Initializing System...")

        if not os.path.exists(model_filename):
            print(f"Error: Model file {model_filename} not found.")
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
        if current_time - self.last_speak_time > 3.0:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
                self.last_speak_time = current_time
            except:
                pass

    def process(self, frame):
        self.frame_count += 1

        # Preprocessing
        resized = cv2.resize(frame, (224, 224))
        normalized = resized.astype('float32') / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)

        # Prediction
        probs = self.model.predict(input_tensor, verbose=0)[0]
        prob_clear = probs[0]
        prob_partial = probs[1]
        prob_block = probs[2]

        threshold = self.thresholds[self.mode]

        status = "GREEN"
        message = "Path Clear"
        color = (0, 255, 0)

        # Logic
        if prob_block > threshold:
            status = "RED"
            message = "STOP! BLOCKED"
            color = (0, 0, 255)
            self.speak("Stop. Way blocked.")

        elif prob_partial > threshold:
            status = "YELLOW"
            message = "WARNING: Obstacle"
            color = (0, 255, 255)
            self.speak("Warning. Obstacle ahead.")

        return frame, status, message, color, probs

    def get_fps(self):
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0

    def switch_mode(self):
        modes = list(self.thresholds.keys())
        current_index = modes.index(self.mode)
        next_index = (current_index + 1) % len(modes)
        self.mode = modes[next_index]
        print(f"Mode switched to: {self.mode}")


def main():
    system = BlindPathSystem()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    print("\n=== System Started ===")
    print("Press 'q' to Quit")
    print("Press 'm' to Switch Mode")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, status, message, color, probs = system.process(frame)

        # UI Drawing
        display = frame.copy()

        cv2.rectangle(display, (0, 0), (640, 120), (0, 0, 0), -1)
        cv2.addWeighted(display[0:120, :], 0.8, frame[0:120, :], 0.2, 0, display[0:120, :])

        cv2.putText(display, f"STATUS: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(display, message, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        info = f"Mode: {system.mode} | FPS: {system.get_fps():.1f} | Clr:{probs[0]:.2f} Prt:{probs[1]:.2f} Blk:{probs[2]:.2f}"
        cv2.putText(display, info, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        h, w = frame.shape[:2]
        cv2.rectangle(display, (w // 4, h // 4), (w * 3 // 4, h * 3 // 4), color, 2)

        cv2.imshow('Blind Path Detection', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            system.switch_mode()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()  