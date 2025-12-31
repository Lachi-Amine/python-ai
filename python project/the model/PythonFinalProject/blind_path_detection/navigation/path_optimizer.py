"""
Path Optimizer for Blind Path Detection System
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class DirectionOption:
    """Direction option"""
    direction: str  # "left", "right", "forward", "backward"
    safety_score: float  # Safety score (0-1)
    effort_score: float  # Effort score (0-1, lower is better)
    distance_score: float  # Distance score (0-1, higher means closer)
    confidence: float  # Confidence


class PathOptimizer:
    """Path optimizer"""

    def __init__(self,
                 safety_weight: float = 0.5,
                 effort_weight: float = 0.3,
                 distance_weight: float = 0.2):
        """
        Initialize path optimizer

        Args:
            safety_weight: Safety weight
            effort_weight: Effort weight
            distance_weight: Distance weight
        """
        self.safety_weight = safety_weight
        self.effort_weight = effort_weight
        self.distance_weight = distance_weight

        # Direction configurations
        self.directions = ["left", "right", "forward", "backward"]

    def evaluate_directions(self,
                            obstacle_map: Dict[str, float],
                            current_heading: float = 0) -> List[DirectionOption]:
        """
        Evaluate all possible directions

        Args:
            obstacle_map: Obstacle map {direction: obstacle_score}
            current_heading: Current heading angle

        Returns:
            List[DirectionOption]: All direction options
        """
        options = []

        for direction in self.directions:
            # Get obstacle score (0 means no obstacle, 1 means fully blocked)
            obstacle_score = obstacle_map.get(direction, 0.0)

            # Calculate safety score (lower obstacle score is safer)
            safety_score = 1.0 - obstacle_score

            # Calculate effort score (turning requires more effort than going straight)
            if direction == "forward":
                effort_score = 0.1
            elif direction == "backward":
                effort_score = 0.9
            else:  # left or right
                effort_score = 0.4

            # Calculate distance score (adjust based on direction preference)
            if direction == "forward":
                distance_score = 1.0
            elif direction in ["left", "right"]:
                distance_score = 0.7
            else:  # backward
                distance_score = 0.3

            # Calculate overall confidence
            confidence = safety_score * 0.7 + (1 - effort_score) * 0.2 + distance_score * 0.1

            option = DirectionOption(
                direction=direction,
                safety_score=safety_score,
                effort_score=effort_score,
                distance_score=distance_score,
                confidence=confidence
            )

            options.append(option)

        return options

    def select_best_direction(self, options: List[DirectionOption]) -> Tuple[str, Dict]:
        """
        Select best direction

        Returns:
            Tuple[str, Dict]: (Best direction, all scores)
        """
        if not options:
            return "forward", {}

        # Calculate weighted total scores
        scores = {}
        for option in options:
            total_score = (
                    self.safety_weight * option.safety_score +
                    self.effort_weight * (1 - option.effort_score) +
                    self.distance_weight * option.distance_score
            )
            scores[option.direction] = total_score

        # Select direction with highest score
        best_direction = max(scores, key=scores.get)

        return best_direction, scores

    def optimize_path(self,
                      obstacle_scores: Dict[str, float],
                      user_preference: str = "safety") -> Dict:
        """
        Optimize path

        Args:
            obstacle_scores: Obstacle scores
            user_preference: User preference ("safety", "speed", "comfort")

        Returns:
            Dict: Optimization result
        """
        # Adjust weights based on user preference
        if user_preference == "speed":
            weights = (0.3, 0.2, 0.5)  # Emphasize distance
        elif user_preference == "comfort":
            weights = (0.4, 0.4, 0.2)  # Emphasize effort
        else:  # safety
            weights = (0.6, 0.3, 0.1)  # Emphasize safety

        self.safety_weight, self.effort_weight, self.distance_weight = weights

        # Evaluate all directions
        options = self.evaluate_directions(obstacle_scores)

        # Select best direction
        best_direction, all_scores = self.select_best_direction(options)

        # Get detailed scores for best option
        best_option = next((opt for opt in options if opt.direction == best_direction), None)

        result = {
            "recommended_direction": best_direction,
            "confidence": best_option.confidence if best_option else 0.0,
            "safety_score": best_option.safety_score if best_option else 0.0,
            "effort_required": best_option.effort_score if best_option else 1.0,
            "all_scores": all_scores,
            "user_preference": user_preference,
            "weights": {
                "safety": self.safety_weight,
                "effort": self.effort_weight,
                "distance": self.distance_weight
            }
        }

        return result