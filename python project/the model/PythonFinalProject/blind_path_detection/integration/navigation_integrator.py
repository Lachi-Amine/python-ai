"""
Navigation Integrator for Blind Path Detection System
"""

import time
from typing import Dict, Any, Optional
import numpy as np

from navigation.path_optimizer import PathOptimizer
from navigation.route_tracker import RouteTracker
from audio.audio_core import AudioCore, SoundLevel
from audio.tts_manager import TTSManager


class NavigationIntegrator:
    """Navigation integrator"""

    def __init__(self,
                 language: str = "en",
                 user_preference: str = "safety"):
        """
        Initialize navigation integrator

        Args:
            language: Language code
            user_preference: User preference ("safety", "speed", "comfort")
        """
        self.language = language
        self.user_preference = user_preference

        # Initialize components
        self.path_optimizer = PathOptimizer()
        self.route_tracker = RouteTracker()
        self.audio_system = AudioCore()
        self.tts_manager = TTSManager()

        # State tracking
        self.current_position = np.array([0.0, 0.0])
        self.current_heading = 0.0
        self.last_obstacle_time = 0
        self.obstacle_cooldown = 2.0
        self.is_off_track = False

        # Obstacle cache
        self.obstacle_cache = {}
        self.cache_timeout = 5.0  # Cache timeout

    def update_position(self,
                        position: np.ndarray,
                        heading: float,
                        expected_heading: Optional[float] = None):
        """
        Update position and heading

        Args:
            position: Current position [x, y]
            heading: Current heading (degrees)
            expected_heading: Expected heading (degrees), optional
        """
        self.current_position = np.array(position)
        self.current_heading = heading % 360

        # Update route tracker
        if expected_heading is not None:
            self.route_tracker.set_expected_heading(expected_heading)

        self.route_tracker.update_current_heading(self.current_heading)

        # Check track status
        self._check_track_status()

    def process_obstacle(self,
                         obstacle_info: Dict[str, Any],
                         model_confidence: float = 1.0):
        """
        Process obstacle detection

        Args:
            obstacle_info: Obstacle information
            model_confidence: Model confidence
        """
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_obstacle_time < self.obstacle_cooldown:
            return

        # Extract obstacle information
        obstacle_type = obstacle_info.get("type", "unknown")
        position = obstacle_info.get("position", [0, 0])
        direction = obstacle_info.get("direction", "forward")
        severity = obstacle_info.get("severity", 0.5)

        # Update cache
        cache_key = f"{obstacle_type}_{direction}"
        self.obstacle_cache[cache_key] = {
            "info": obstacle_info,
            "timestamp": current_time
        }

        # Clean expired cache
        self._clean_cache(current_time)

        # Calculate best evasion direction
        best_direction = self._calculate_evasion(obstacle_info)

        # Trigger audio alert
        self._trigger_audio_alert(obstacle_type, direction, model_confidence)

        # Provide navigation guidance if needed
        if obstacle_type in ["partial", "full"]:
            self._provide_navigation_guidance(best_direction)

        self.last_obstacle_time = current_time

    def _calculate_evasion(self, obstacle_info: Dict[str, Any]) -> str:
        """Calculate best evasion direction"""
        # Create obstacle map
        obstacle_scores = self._create_obstacle_scores(obstacle_info)

        # Optimize path
        result = self.path_optimizer.optimize_path(
            obstacle_scores,
            self.user_preference
        )

        return result["recommended_direction"]

    def _create_obstacle_scores(self, obstacle_info: Dict[str, Any]) -> Dict[str, float]:
        """Create obstacle score map"""
        # Get all obstacles from cache
        current_time = time.time()
        obstacle_scores = {d: 0.0 for d in ["forward", "right", "backward", "left"]}

        for cache_key, cache_data in self.obstacle_cache.items():
            # Check if expired
            if current_time - cache_data["timestamp"] > self.cache_timeout:
                continue

            info = cache_data["info"]
            direction = info.get("direction", "forward")
            severity = info.get("severity", 0.5)

            # Accumulate scores
            obstacle_scores[direction] += severity

        # Limit to 0-1 range
        for direction in obstacle_scores:
            obstacle_scores[direction] = min(1.0, obstacle_scores[direction])

        return obstacle_scores

    def _trigger_audio_alert(self, obstacle_type: str, direction: str, confidence: float):
        """Trigger audio alert"""
        # Map obstacle type to sound level
        if obstacle_type == "clear":
            sound_level = SoundLevel.GREEN
        elif obstacle_type == "partial":
            sound_level = SoundLevel.YELLOW
        elif obstacle_type == "full":
            sound_level = SoundLevel.RED
        else:
            sound_level = SoundLevel.LOW_CONFIDENCE

        # Calculate pan based on direction
        pan = self._direction_to_pan(direction)

        # Adjust volume based on confidence
        volume_factor = min(1.0, confidence * 1.5)

        # Play sound
        try:
            self.audio_system.play_sound_with_pan(sound_level, pan)
        except Exception as e:
            print(f"Audio error: {e}")

    def _provide_navigation_guidance(self, recommended_direction: str):
        """Provide navigation guidance"""
        # Simple direction guidance
        if recommended_direction == "left":
            guidance = "Please move to the left."
        elif recommended_direction == "right":
            guidance = "Please move to the right."
        elif recommended_direction == "backward":
            guidance = "Please step back carefully."
        else:
            guidance = "Please proceed with caution."

        # Integrate with TTS
        print(f"Navigation guidance: {guidance}")
        # self.tts_manager.speak(guidance, self.language)

    def _direction_to_pan(self, direction: str) -> float:
        """Convert direction to pan"""
        if direction == "left":
            return -0.8
        elif direction == "right":
            return 0.8
        else:
            return 0.0

    def _check_track_status(self):
        """Check route status"""
        on_track, angle_diff, deviation_duration = self.route_tracker.get_deviation_info()

        if not on_track and not self.is_off_track:
            # Just deviated from route
            self.is_off_track = True
            self._handle_off_track(angle_diff)
        elif on_track and self.is_off_track:
            # Returned to route
            self.is_off_track = False
            self._handle_return_to_track()

    def _handle_off_track(self, angle_diff: float):
        """Handle off-track situation"""
        print(f"Warning: Off track! Angle difference: {angle_diff:.1f} degrees")

        # Get correction direction
        correction_dir = self.route_tracker.get_correction_direction()

        # Audio guidance
        guidance = f"You are off track. Please turn {correction_dir}."
        print(f"Audio guidance: {guidance}")
        # self.tts_manager.speak(guidance, self.language)

    def _handle_return_to_track(self):
        """Handle return to track"""
        print("Info: Returned to correct route")
        # Can add return confirmation prompt

    def _clean_cache(self, current_time: float):
        """Clean expired cache"""
        expired_keys = [
            key for key, data in self.obstacle_cache.items()
            if current_time - data["timestamp"] > self.cache_timeout
        ]

        for key in expired_keys:
            del self.obstacle_cache[key]

    def set_language(self, language: str):
        """Set language"""
        self.language = language
        self.tts_manager.set_language(language)

    def set_user_preference(self, preference: str):
        """Set user preference"""
        if preference in ["safety", "speed", "comfort"]:
            self.user_preference = preference

    def cleanup(self):
        """Cleanup resources"""
        self.audio_system.cleanup()