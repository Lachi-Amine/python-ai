"""
Spatial Mapper for Blind Path Detection System
"""

import numpy as np
from typing import Dict, List


class SpatialMapper:
    """Spatial mapper"""

    def __init__(self):
        # Direction to angle mapping
        self.direction_angles = {
            "forward": 0,
            "right": 90,
            "backward": 180,
            "left": 270
        }

        # Stereo pan mapping
        self.pan_mapping = {
            "left": -1.0,
            "center-left": -0.6,
            "center": 0.0,
            "center-right": 0.6,
            "right": 1.0
        }

    def map_obstacle_to_direction(self,
                                  obstacle_position: np.ndarray,
                                  user_position: np.ndarray,
                                  user_heading: float) -> Dict:
        """
        Map obstacle position to direction

        Args:
            obstacle_position: Obstacle coordinates [x, y]
            user_position: User coordinates [x, y]
            user_heading: User heading (degrees)

        Returns:
            Dict: Direction mapping result
        """
        # Calculate relative position
        rel_x = obstacle_position[0] - user_position[0]
        rel_y = obstacle_position[1] - user_position[1]

        # Calculate absolute angle
        absolute_angle = np.degrees(np.arctan2(rel_y, rel_x)) % 360

        # Calculate angle relative to user heading
        relative_angle = (absolute_angle - user_heading) % 360

        # Map to direction
        direction = self._angle_to_direction(relative_angle)

        # Calculate distance
        distance = np.sqrt(rel_x ** 2 + rel_y ** 2)

        # Calculate pan
        pan = self._direction_to_pan(direction)

        return {
            "direction": direction,
            "relative_angle": relative_angle,
            "distance": distance,
            "pan": pan,
            "absolute_position": {
                "x": obstacle_position[0],
                "y": obstacle_position[1]
            }
        }

    def _angle_to_direction(self, angle: float) -> str:
        """Convert angle to direction"""
        if 315 <= angle <= 360 or 0 <= angle < 45:
            return "forward"
        elif 45 <= angle < 135:
            return "right"
        elif 135 <= angle < 225:
            return "backward"
        else:  # 225 <= angle < 315
            return "left"

    def _direction_to_pan(self, direction: str) -> float:
        """Convert direction to pan"""
        if direction == "left":
            return -0.8
        elif direction == "right":
            return 0.8
        else:
            return 0.0

    def create_obstacle_map(self,
                            obstacles: List[Dict],
                            user_position: np.ndarray,
                            user_heading: float) -> Dict[str, float]:
        """
        Create obstacle map

        Args:
            obstacles: Obstacle list
            user_position: User position
            user_heading: User heading

        Returns:
            Dict[str, float]: Obstacle score for each direction
        """
        direction_scores = {d: 0.0 for d in ["forward", "right", "backward", "left"]}

        for obstacle in obstacles:
            # Get obstacle information
            position = obstacle.get("position", [0, 0])
            severity = obstacle.get("severity", 0.5)  # Severity 0-1
            distance = obstacle.get("distance", 1.0)

            # Map to direction
            mapping = self.map_obstacle_to_direction(position, user_position, user_heading)
            direction = mapping["direction"]

            # Adjust score based on distance (closer means higher score)
            distance_factor = max(0, 1 - distance / 10)  # Assume effective within 10 meters
            score = severity * distance_factor

            # Accumulate score
            direction_scores[direction] += score

        # Normalize to 0-1 range
        max_score = max(direction_scores.values()) if direction_scores else 1.0
        if max_score > 0:
            for direction in direction_scores:
                direction_scores[direction] = min(1.0, direction_scores[direction] / max_score)

        return direction_scores