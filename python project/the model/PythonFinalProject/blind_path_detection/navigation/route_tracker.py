"""
Route Tracker for Blind Path Detection System
"""

import time
from typing import Optional, Tuple


class RouteTracker:
    """Route tracker"""

    def __init__(self,
                 tolerance_angle: float = 15.0,
                 max_deviation_time: float = 3.0,
                 update_interval: float = 0.5):
        """
        Initialize route tracker

        Args:
            tolerance_angle: Allowed deviation angle (degrees)
            max_deviation_time: Maximum deviation time (seconds)
            update_interval: Update interval (seconds)
        """
        self.tolerance_angle = tolerance_angle
        self.max_deviation_time = max_deviation_time
        self.update_interval = update_interval

        self.expected_heading = None
        self.current_heading = None
        self.deviation_start_time = None
        self.is_on_track = True
        self.last_update_time = 0

        # History trajectory
        self.heading_history = []
        self.time_history = []

    def set_expected_heading(self, heading: float):
        """Set expected heading"""
        self.expected_heading = heading % 360

    def update_current_heading(self, heading: float):
        """Update current heading"""
        current_time = time.time()

        # Control update frequency
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time
        self.current_heading = heading % 360

        # Record history
        self.heading_history.append(self.current_heading)
        self.time_history.append(current_time)

        # Maintain history length
        if len(self.heading_history) > 100:
            self.heading_history.pop(0)
            self.time_history.pop(0)

        # Check deviation
        self._check_deviation()

    def _check_deviation(self):
        """Check if deviated from route"""
        if self.expected_heading is None or self.current_heading is None:
            return

        # Calculate angle difference
        angle_diff = self._calculate_angle_difference(
            self.expected_heading,
            self.current_heading
        )

        # Check if exceeds tolerance range
        if abs(angle_diff) <= self.tolerance_angle:
            # On correct route
            self.is_on_track = True
            self.deviation_start_time = None
        else:
            # Deviated from route
            if self.deviation_start_time is None:
                self.deviation_start_time = time.time()

            # Check if deviated for too long
            deviation_duration = time.time() - self.deviation_start_time
            if deviation_duration > self.max_deviation_time:
                self.is_on_track = False
            else:
                self.is_on_track = True

    def _calculate_angle_difference(self, angle1: float, angle2: float) -> float:
        """Calculate minimum difference between two angles"""
        diff = angle1 - angle2
        diff = ((diff + 180) % 360) - 180
        return diff

    def get_deviation_info(self) -> Tuple[bool, float, Optional[float]]:
        """
        Get deviation information

        Returns:
            Tuple[bool, float, Optional[float]]:
            (Whether on route, Angle difference, Deviation duration)
        """
        if self.expected_heading is None or self.current_heading is None:
            return True, 0.0, None

        angle_diff = self._calculate_angle_difference(
            self.expected_heading,
            self.current_heading
        )

        deviation_duration = None
        if self.deviation_start_time is not None:
            deviation_duration = time.time() - self.deviation_start_time

        return self.is_on_track, angle_diff, deviation_duration

    def get_correction_direction(self) -> str:
        """Get correction direction"""
        _, angle_diff, _ = self.get_deviation_info()

        if angle_diff > 0:
            return "right"  # Need to correct right
        else:
            return "left"  # Need to correct left

    def reset(self):
        """Reset tracker"""
        self.deviation_start_time = None
        self.is_on_track = True
        self.heading_history.clear()
        self.time_history.clear()