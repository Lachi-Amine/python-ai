"""
Audio Core Engine for Blind Path Detection System
"""

import numpy as np
import pyaudio
import threading
import time
from enum import Enum
from typing import Dict, Optional


class SoundLevel(Enum):
    """Sound level enumeration"""
    GREEN = 1  # Safe, mild alert
    YELLOW = 2  # Warning, medium alert
    RED = 3  # Danger, strong alert
    LOW_CONFIDENCE = 4  # Low confidence, special alert


class AudioCore:
    """Audio core engine"""

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_playing = False

        # Sound configurations
        self.sound_profiles: Dict[SoundLevel, Dict] = {
            SoundLevel.GREEN: {
                "frequency": 800,
                "duration": 0.1,
                "interval": None,  # Don't repeat
                "volume": 0.3,
                "wave_type": "sine"
            },
            SoundLevel.YELLOW: {
                "frequency": 1000,
                "duration": 0.15,
                "interval": 1.0,
                "volume": 0.6,
                "wave_type": "square"
            },
            SoundLevel.RED: {
                "frequency": 1200,
                "duration": 0.2,
                "interval": 0.5,
                "volume": 0.9,
                "wave_type": "square"
            },
            SoundLevel.LOW_CONFIDENCE: {
                "frequency": 600,
                "duration": 0.1,
                "interval": 0.8,
                "volume": 0.5,
                "wave_type": "sine"
            }
        }

    def generate_tone(self, frequency, duration, volume=0.5, wave_type="sine"):
        """Generate tone"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)

        if wave_type == "sine":
            tone = np.sin(2 * np.pi * frequency * t)
        elif wave_type == "square":
            tone = np.sign(np.sin(2 * np.pi * frequency * t))
        elif wave_type == "sawtooth":
            tone = 2 * (t * frequency - np.floor(0.5 + t * frequency))
        else:
            tone = np.sin(2 * np.pi * frequency * t)

        # Apply fade in/out
        fade_samples = int(self.sample_rate * 0.01)
        if len(tone) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out

        # Apply volume
        tone = tone * volume

        # Convert to stereo
        stereo = np.column_stack((tone, tone))
        return stereo.astype(np.float32)

    def play_sound_with_pan(self, level: SoundLevel, pan=0.0):
        """Play sound with pan control"""
        if level not in self.sound_profiles:
            raise ValueError(f"Unknown sound level: {level}")

        profile = self.sound_profiles[level]

        # Generate tone
        audio_data = self.generate_tone(
            frequency=profile["frequency"],
            duration=profile["duration"],
            volume=profile["volume"],
            wave_type=profile["wave_type"]
        )

        # Apply pan control
        if pan != 0.0:
            audio_data[:, 0] *= max(0, 1 - pan)  # Left channel
            audio_data[:, 1] *= max(0, 1 + pan)  # Right channel

        # Play audio
        self._play_audio_data(audio_data)

        # Repeat if needed
        if profile["interval"] is not None:
            threading.Thread(
                target=self._repeat_sound,
                args=(level, pan, profile["interval"]),
                daemon=True
            ).start()

    def _play_audio_data(self, audio_data):
        """Play audio data"""
        if self.stream is None:
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=2,
                rate=self.sample_rate,
                output=True
            )

        self.is_playing = True
        self.stream.write(audio_data.tobytes())
        self.is_playing = False

    def _repeat_sound(self, level, pan, interval):
        """Repeat sound"""
        while self.is_playing:
            time.sleep(interval)
            self.play_sound_with_pan(level, pan)

    def cleanup(self):
        """Cleanup resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()