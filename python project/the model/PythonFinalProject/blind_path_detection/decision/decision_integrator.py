"""
Decision Integrator for Blind Path Detection System
"""

import numpy as np
from typing import Dict, Any
from config import THRESHOLDS, CLASS_NAMES


class DecisionIntegrator:
    """Decision integrator"""

    def __init__(self,
                 safety_mode: str = "conservative",
                 enable_confidence_check: bool = True,
                 min_confidence: float = 0.5):
        """
        Initialize decision integrator

        Args:
            safety_mode: Safety mode ("conservative", "balanced", "aggressive")
            enable_confidence_check: Whether to enable confidence checking
            min_confidence: Minimum confidence threshold
        """
        self.safety_mode = safety_mode
        self.enable_confidence_check = enable_confidence_check
        self.min_confidence = min_confidence

        # Get threshold configuration
        self.thresholds = THRESHOLDS.get(safety_mode, THRESHOLDS["conservative"])

        # Decision history (for smoothing)
        self.decision_history = []
        self.max_history_size = 5

    def make_decision(self,
                      probabilities: np.ndarray,
                      confidence: float = None,
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        修正后的决策逻辑：优先检查危险，而不是只看最大概率
        假设 probabilities 顺序: [0: Clear, 1: Partial, 2: Block]
        """

        # 1. 获取各个类别的概率
        # 注意：如果你的模型输出顺序不同，请在这里调整索引
        prob_clear = probabilities[0]
        prob_partial = probabilities[1]
        prob_block = probabilities[2]

        # 2. 获取当前模式下的阈值 (从 self.thresholds 获取)
        # 如果字典里没有 key，就用默认值: Block=0.4, Partial=0.45
        thresh_block = self.thresholds.get('block_threshold', 0.4)
        thresh_partial = self.thresholds.get('partial_threshold', 0.45)

        # === 核心修复逻辑 ===

        # 判定优先级 1: 只要完全遮挡(Block)分数超过阈值，直接停
        if prob_block > thresh_block:
            final_class = "Block"
            status = "STOP"
            color = "RED"

        # 判定优先级 2: 如果部分遮挡(Partial)分数高，也要根据情况报警
        # (你之前的问题就是漏掉了这里，或者这里的阈值太高了)
        elif prob_partial > thresh_partial:
            final_class = "Partial"
            status = "WARNING"  # 或者 "AVOID"
            color = "YELLOW"  # 界面会显示黄色

        # 判定优先级 3: 只有在以上都不满足时，才认为是安全的
        else:
            final_class = "Clear"
            status = "Go"
            color = "GREEN"

        # 3. 返回决策结果字典
        return {
            "status": status,  # 显示在屏幕上的大字 (STOP / WARNING / Go)
            "color": color,  # 颜色
            "class_name": final_class,  # 类别名
            "confidence": float(np.max(probabilities)),  # 置信度
            "action": status  # 动作指令
        }

    def _threshold_decision(self,
                            clear_prob: float,
                            partial_prob: float,
                            full_prob: float,
                            confidence: float) -> Dict[str, Any]:
        """Threshold-based decision"""

        # Get thresholds
        full_threshold = self.thresholds["full"]
        partial_threshold = self.thresholds["partial"]
        clear_threshold = self.thresholds["clear"]
        confidence_threshold = self.thresholds["confidence"]

        # Check confidence
        if confidence < confidence_threshold:
            return {
                "level": "LOW_CONFIDENCE",
                "message": "Low confidence detection. Please proceed with caution.",
                "action": "caution",
                "risk_level": "medium",
                "audio_level": "warning"
            }

        # High risk: Fully blocked
        if full_prob >= full_threshold:
            return {
                "level": "RED",
                "message": "Path fully blocked. Stop immediately!",
                "action": "stop",
                "risk_level": "high",
                "audio_level": "danger",
                "obstacle_type": "full"
            }

        # Medium risk: Partially blocked
        if partial_prob >= partial_threshold:
            return {
                "level": "YELLOW",
                "message": "Path partially blocked. Adjust direction.",
                "action": "adjust",
                "risk_level": "medium",
                "audio_level": "warning",
                "obstacle_type": "partial"
            }

        # Low risk: Clear
        if clear_prob >= clear_threshold:
            return {
                "level": "GREEN",
                "message": "Path clear. Continue forward.",
                "action": "go",
                "risk_level": "low",
                "audio_level": "info",
                "obstacle_type": "clear"
            }

        # Default: Uncertain
        return {
            "level": "LOW_CONFIDENCE",
            "message": "Uncertain path condition. Proceed carefully.",
            "action": "caution",
            "risk_level": "medium",
            "audio_level": "warning"
        }

    def _low_confidence_decision(self,
                                 confidence: float,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Low confidence decision"""
        # Adjust decision based on context
        has_previous_obstacle = context.get("has_previous_obstacle", False)

        if has_previous_obstacle:
            message = "Continuing obstacle alert. Use extreme caution."
            risk_level = "high"
        else:
            message = f"Low confidence detection ({confidence:.2f}). Please proceed with caution."
            risk_level = "medium"

        return {
            "level": "LOW_CONFIDENCE",
            "message": message,
            "action": "caution",
            "risk_level": risk_level,
            "audio_level": "warning",
            "confidence": float(confidence)
        }

    def _record_decision(self, decision: Dict[str, Any]):
        """Record decision history"""
        self.decision_history.append(decision)

        # Limit history size
        if len(self.decision_history) > self.max_history_size:
            self.decision_history.pop(0)

    def _apply_smoothing(self) -> Dict[str, Any]:
        """Apply history smoothing"""
        if len(self.decision_history) < 2:
            return None

        # Statistics for recent decisions
        level_counts = {}
        risk_levels = {}

        for decision in self.decision_history:
            level = decision.get("level", "UNKNOWN")
            risk = decision.get("risk_level", "medium")

            level_counts[level] = level_counts.get(level, 0) + 1
            risk_levels[risk] = risk_levels.get(risk, 0) + 1

        # Find most frequent decision
        most_common_level = max(level_counts, key=level_counts.get)
        most_common_risk = max(risk_levels, key=risk_levels.get)

        # Get latest decision as base
        latest_decision = self.decision_history[-1]

        # Apply smoothing if stable pattern exists
        if level_counts[most_common_level] >= len(self.decision_history) * 0.6:
            smoothed_decision = latest_decision.copy()
            smoothed_decision["level"] = most_common_level
            smoothed_decision["risk_level"] = most_common_risk
            smoothed_decision["smoothed"] = True

            return smoothed_decision

        return None

    def update_safety_mode(self, mode: str):
        """Update safety mode"""
        if mode in THRESHOLDS:
            self.safety_mode = mode
            self.thresholds = THRESHOLDS[mode]
            return True
        return False