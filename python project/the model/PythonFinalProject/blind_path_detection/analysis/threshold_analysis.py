"""
Threshold Analysis for Blind Path Detection System

This module analyzes the impact of different decision thresholds on safety and performance.
It helps find the optimal thresholds for different safety modes.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import pandas as pd
from config import *


class ThresholdAnalyzer:
    """Analyzes different threshold settings for decision making"""

    def __init__(self, model, validation_data):
        """
        Initialize threshold analyzer

        Args:
            model: Trained model
            validation_data: Validation dataset
        """
        self.model = model
        self.validation_data = validation_data

        # Collect predictions and labels
        self.y_true, self.y_pred, self.probs = self._collect_predictions()

        # Default thresholds for different modes
        self.default_thresholds = {
            "conservative": {
                "full": 0.5,
                "partial": 0.4,
                "clear": 0.75,
                "confidence": 0.5
            },
            "balanced": {
                "full": 0.45,
                "partial": 0.35,
                "clear": 0.7,
                "confidence": 0.45
            },
            "aggressive": {
                "full": 0.35,
                "partial": 0.3,
                "clear": 0.6,
                "confidence": 0.4
            }
        }

    def _collect_predictions(self):
        """Collect predictions and true labels from validation data"""
        y_true = []
        y_pred = []
        probs = []

        for batch_images, batch_labels in self.validation_data:
            batch_probs = self.model.predict(batch_images, verbose=0)
            batch_preds = np.argmax(batch_probs, axis=1)
            batch_true = np.argmax(batch_labels.numpy(), axis=1)

            y_true.extend(batch_true)
            y_pred.extend(batch_preds)
            probs.extend(batch_probs)

        return np.array(y_true), np.array(y_pred), np.array(probs)

    def analyze_threshold_impact(self, thresholds=None):
        """
        Analyze impact of different thresholds

        Args:
            thresholds: List of threshold values to analyze

        Returns:
            dict: Analysis results for each threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0.3, 0.9, 13)

        results = []

        for threshold in thresholds:
            # Apply threshold-based decision
            decisions = self._apply_threshold_decision(threshold)

            # Calculate metrics
            metrics = self._calculate_metrics(decisions)
            metrics['threshold'] = threshold

            results.append(metrics)

        return results

    def _apply_threshold_decision(self, threshold):
        """
        Apply threshold-based decision to all samples

        Args:
            threshold: Decision threshold

        Returns:
            list: Decision for each sample
        """
        decisions = []

        for probs in self.probs:
            clear_prob, partial_prob, full_prob = probs

            if full_prob >= threshold:
                decisions.append(2)  # Fully Blocked
            elif partial_prob >= threshold:
                decisions.append(1)  # Partially Blocked
            elif clear_prob >= threshold:
                decisions.append(0)  # Clear
            else:
                decisions.append(0)  # Default to Clear (low confidence)

        return np.array(decisions)

    def _calculate_metrics(self, decisions):
        """
        Calculate safety and performance metrics

        Args:
            decisions: Threshold-based decisions

        Returns:
            dict: Calculated metrics
        """
        # Confusion matrix
        cm = confusion_matrix(self.y_true, decisions)

        # Calculate safety metrics
        safety_metrics = self._calculate_safety_metrics(self.y_true, decisions)

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(self.y_true, decisions)

        # Combine all metrics
        metrics = {
            **safety_metrics,
            **performance_metrics,
            "confusion_matrix": cm,
            "total_samples": len(self.y_true),
            "false_alarms": np.sum((decisions > 0) & (self.y_true == 0)),
            "missed_detections": np.sum((decisions == 0) & (self.y_true > 0))
        }

        return metrics

    def _calculate_safety_metrics(self, y_true, decisions):
        """Calculate safety-related metrics"""
        # High-risk errors: Blocked misclassified as Clear
        high_risk_errors = np.sum((y_true > 0) & (decisions == 0))

        # Medium-risk errors: Severity misclassification
        medium_risk_errors = np.sum(
            ((y_true == 1) & (decisions == 2)) |  # Partial -> Full
            ((y_true == 2) & (decisions == 1))  # Full -> Partial
        )

        # Low-risk errors: Clear misclassified as Blocked
        low_risk_errors = np.sum((y_true == 0) & (decisions > 0))

        # Safety score (higher is safer)
        total_samples = len(y_true)
        safety_score = 1.0 - (high_risk_errors * 3 + medium_risk_errors * 2 + low_risk_errors) / (total_samples * 3)

        return {
            "high_risk_errors": int(high_risk_errors),
            "medium_risk_errors": int(medium_risk_errors),
            "low_risk_errors": int(low_risk_errors),
            "safety_score": float(safety_score),
            "high_risk_percentage": float(high_risk_errors / total_samples * 100) if total_samples > 0 else 0
        }

    def _calculate_performance_metrics(self, y_true, decisions):
        """Calculate performance metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # Overall metrics
        accuracy = accuracy_score(y_true, decisions)

        # Per-class metrics
        precision_macro = precision_score(y_true, decisions, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, decisions, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, decisions, average='macro', zero_division=0)

        # Per-class detailed metrics
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []

        for i in range(NUM_CLASSES):
            precision = precision_score(y_true == i, decisions == i, zero_division=0)
            recall = recall_score(y_true == i, decisions == i, zero_division=0)
            f1 = f1_score(y_true == i, decisions == i, zero_division=0)

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)

        return {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_per_class": [float(p) for p in precision_per_class],
            "recall_per_class": [float(r) for r in recall_per_class],
            "f1_per_class": [float(f) for f in f1_per_class]
        }

    def find_optimal_thresholds(self, safety_weight=0.7, performance_weight=0.3):
        """
        Find optimal thresholds based on weighted objectives

        Args:
            safety_weight: Weight for safety metrics (0-1)
            performance_weight: Weight for performance metrics (0-1)

        Returns:
            dict: Optimal threshold and analysis
        """
        # Analyze thresholds from 0.3 to 0.9
        thresholds = np.linspace(0.3, 0.9, 13)
        results = self.analyze_threshold_impact(thresholds)

        # Calculate combined scores
        combined_scores = []

        for result in results:
            # Normalize metrics to 0-1 range
            safety_norm = result['safety_score']
            performance_norm = result['accuracy']

            # Combined weighted score
            combined_score = (safety_weight * safety_norm +
                              performance_weight * performance_norm)

            combined_scores.append(combined_score)

        # Find optimal threshold
        optimal_idx = np.argmax(combined_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_result = results[optimal_idx]

        return {
            "optimal_threshold": float(optimal_threshold),
            "combined_score": float(combined_scores[optimal_idx]),
            "safety_score": optimal_result['safety_score'],
            "accuracy": optimal_result['accuracy'],
            "analysis": optimal_result,
            "all_results": results
        }

    def compare_safety_modes(self):
        """Compare performance of different safety modes"""
        comparisons = []

        for mode_name, thresholds in self.default_thresholds.items():
            # Apply multi-threshold decision
            decisions = self._apply_multi_threshold_decision(thresholds)

            # Calculate metrics
            safety_metrics = self._calculate_safety_metrics(self.y_true, decisions)
            performance_metrics = self._calculate_performance_metrics(self.y_true, decisions)

            comparison = {
                "mode": mode_name,
                "thresholds": thresholds,
                **safety_metrics,
                **performance_metrics
            }

            comparisons.append(comparison)

        return comparisons

    def _apply_multi_threshold_decision(self, thresholds):
        """Apply multi-threshold decision"""
        decisions = []

        for probs in self.probs:
            clear_prob, partial_prob, full_prob = probs

            if full_prob >= thresholds['full']:
                decisions.append(2)  # Fully Blocked
            elif partial_prob >= thresholds['partial']:
                decisions.append(1)  # Partially Blocked
            elif clear_prob >= thresholds['clear']:
                decisions.append(0)  # Clear
            else:
                # Low confidence
                if full_prob > partial_prob and full_prob > clear_prob:
                    decisions.append(2)
                elif partial_prob > clear_prob:
                    decisions.append(1)
                else:
                    decisions.append(0)

        return np.array(decisions)

    def visualize_threshold_analysis(self, save_path=None):
        """Create visualization of threshold analysis"""
        if save_path is None:
            save_path = OUTPUT_DIR / "threshold_analysis.png"

        # Get threshold analysis results
        thresholds = np.linspace(0.3, 0.9, 13)
        results = self.analyze_threshold_impact(thresholds)

        # Extract metrics
        threshold_values = [r['threshold'] for r in results]
        accuracy_scores = [r['accuracy'] for r in results]
        safety_scores = [r['safety_score'] for r in results]
        high_risk_errors = [r['high_risk_errors'] for r in results]
        false_alarms = [r['false_alarms'] for r in results]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Accuracy vs Safety
        ax1 = axes[0, 0]
        ax1.plot(threshold_values, accuracy_scores, 'b-o', label='Accuracy', linewidth=2)
        ax1.plot(threshold_values, safety_scores, 'r-s', label='Safety Score', linewidth=2)
        ax1.set_xlabel('Decision Threshold', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Accuracy vs Safety Score', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Highlight optimal region
        optimal_idx = np.argmax([a * 0.3 + s * 0.7 for a, s in zip(accuracy_scores, safety_scores)])
        ax1.axvline(x=threshold_values[optimal_idx], color='green', linestyle='--',
                    label=f'Optimal: {threshold_values[optimal_idx]:.2f}')
        ax1.legend()

        # Plot 2: High-risk errors
        ax2 = axes[0, 1]
        ax2.plot(threshold_values, high_risk_errors, 'r-o', linewidth=2)
        ax2.set_xlabel('Decision Threshold', fontsize=12)
        ax2.set_ylabel('High-risk Errors', fontsize=12)
        ax2.set_title('High-risk Errors vs Threshold', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(threshold_values, 0, high_risk_errors, alpha=0.3, color='red')

        # Plot 3: False alarms
        ax3 = axes[1, 0]
        ax3.plot(threshold_values, false_alarms, 'g-o', linewidth=2)
        ax3.set_xlabel('Decision Threshold', fontsize=12)
        ax3.set_ylabel('False Alarms', fontsize=12)
        ax3.set_title('False Alarms vs Threshold', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.fill_between(threshold_values, 0, false_alarms, alpha=0.3, color='green')

        # Plot 4: Trade-off analysis
        ax4 = axes[1, 1]
        scatter = ax4.scatter(accuracy_scores, safety_scores, c=threshold_values,
                              cmap='viridis', s=100, alpha=0.7)
        ax4.set_xlabel('Accuracy', fontsize=12)
        ax4.set_ylabel('Safety Score', fontsize=12)
        ax4.set_title('Accuracy-Safety Trade-off', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Add threshold labels
        for i, (acc, safe, thresh) in enumerate(zip(accuracy_scores, safety_scores, threshold_values)):
            if i % 2 == 0:  # Label every other point
                ax4.annotate(f'{thresh:.2f}', (acc, safe), fontsize=9)

        # Add colorbar
        plt.colorbar(scatter, ax=ax4, label='Threshold Value')

        # Add optimal point
        optimal_acc = accuracy_scores[optimal_idx]
        optimal_safe = safety_scores[optimal_idx]
        ax4.scatter(optimal_acc, optimal_safe, c='red', s=200, marker='*',
                    label=f'Optimal (t={threshold_values[optimal_idx]:.2f})')
        ax4.legend()

        plt.suptitle('Threshold Analysis for Decision Making', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save figure
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Threshold analysis visualization saved to: {save_path}")

        return fig

    def generate_report(self, save_path=None):
        """Generate comprehensive threshold analysis report"""
        if save_path is None:
            save_path = OUTPUT_DIR / "threshold_analysis_report.txt"

        # Get optimal thresholds
        optimal_result = self.find_optimal_thresholds()

        # Compare safety modes
        mode_comparisons = self.compare_safety_modes()

        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BLIND PATH DETECTION SYSTEM - THRESHOLD ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. EXECUTIVE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Optimal Decision Threshold: {optimal_result['optimal_threshold']:.3f}\n")
            f.write(f"Safety Score at Optimal: {optimal_result['safety_score']:.4f}\n")
            f.write(f"Accuracy at Optimal: {optimal_result['accuracy']:.4f}\n")
            f.write(f"Combined Score: {optimal_result['combined_score']:.4f}\n\n")

            f.write("2. SAFETY MODE COMPARISON\n")
            f.write("-" * 80 + "\n")

            # Create comparison table
            f.write(f"{'Mode':<15} {'Accuracy':<12} {'Safety Score':<15} {'High-risk':<12} {'False Alarms':<15}\n")
            f.write("-" * 80 + "\n")

            for comp in mode_comparisons:
                f.write(f"{comp['mode']:<15} {comp['accuracy']:<12.4f} ")
                f.write(f"{comp['safety_score']:<15.4f} {comp['high_risk_errors']:<12} ")
                f.write(f"{comp['false_alarms']:<15}\n")

            f.write("\n3. DETAILED THRESHOLD ANALYSIS\n")
            f.write("-" * 80 + "\n")

            # Analyze range of thresholds
            thresholds = np.linspace(0.3, 0.9, 7)
            results = self.analyze_threshold_impact(thresholds)

            f.write(f"{'Threshold':<12} {'Accuracy':<12} {'Safety':<12} {'High-risk':<12} ")
            f.write(f"{'Medium-risk':<12} {'Low-risk':<12} {'False Alarms':<15}\n")
            f.write("-" * 80 + "\n")

            for result in results:
                f.write(f"{result['threshold']:<12.3f} {result['accuracy']:<12.4f} ")
                f.write(f"{result['safety_score']:<12.4f} {result['high_risk_errors']:<12} ")
                f.write(f"{result['medium_risk_errors']:<12} {result['low_risk_errors']:<12} ")
                f.write(f"{result['false_alarms']:<15}\n")

            f.write("\n4. RISK ANALYSIS\n")
            f.write("-" * 80 + "\n")

            # Calculate risk at optimal threshold
            optimal_decisions = self._apply_threshold_decision(optimal_result['optimal_threshold'])
            risk_metrics = self._calculate_safety_metrics(self.y_true, optimal_decisions)

            total_samples = len(self.y_true)
            f.write(f"Total Samples: {total_samples}\n")
            f.write(f"High-risk Errors: {risk_metrics['high_risk_errors']} ")
            f.write(f"({risk_metrics['high_risk_percentage']:.2f}%)\n")
            f.write(f"Medium-risk Errors: {risk_metrics['medium_risk_errors']}\n")
            f.write(f"Low-risk Errors: {risk_metrics['low_risk_errors']}\n")

            # Risk assessment
            high_risk_percentage = risk_metrics['high_risk_percentage']
            if high_risk_percentage < 1.0:
                f.write("\n✅ LOW RISK: System has minimal high-risk errors.\n")
            elif high_risk_percentage < 5.0:
                f.write("\n⚠️ MODERATE RISK: Acceptable level of high-risk errors.\n")
            else:
                f.write("\n❌ HIGH RISK: Too many high-risk errors. Consider more conservative threshold.\n")

            f.write("\n5. RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")

            # Generate recommendations based on analysis
            if optimal_result['optimal_threshold'] < 0.4:
                f.write("Recommended Safety Mode: AGGRESSIVE\n")
                f.write("  - Use when speed is more important than absolute safety\n")
                f.write("  - Suitable for well-maintained, familiar paths\n")
                f.write("  - Higher false alarm rate but faster detection\n")
            elif optimal_result['optimal_threshold'] < 0.55:
                f.write("Recommended Safety Mode: BALANCED\n")
                f.write("  - Good balance between safety and performance\n")
                f.write("  - Suitable for general use\n")
                f.write("  - Moderate false alarm rate with good safety\n")
            else:
                f.write("Recommended Safety Mode: CONSERVATIVE\n")
                f.write("  - Maximum safety, minimize high-risk errors\n")
                f.write("  - Suitable for unfamiliar or dangerous paths\n")
                f.write("  - Higher false alarm rate but minimal risk\n")

            f.write("\n6. THRESHOLD SETTINGS FOR DEPLOYMENT\n")
            f.write("-" * 80 + "\n")

            f.write("Conservative Mode (Maximum Safety):\n")
            f.write(f"  Full Blocked Threshold: {self.default_thresholds['conservative']['full']:.2f}\n")
            f.write(f"  Partial Blocked Threshold: {self.default_thresholds['conservative']['partial']:.2f}\n")
            f.write(f"  Clear Threshold: {self.default_thresholds['conservative']['clear']:.2f}\n")
            f.write(f"  Confidence Threshold: {self.default_thresholds['conservative']['confidence']:.2f}\n\n")

            f.write("Balanced Mode (General Use):\n")
            f.write(f"  Full Blocked Threshold: {self.default_thresholds['balanced']['full']:.2f}\n")
            f.write(f"  Partial Blocked Threshold: {self.default_thresholds['balanced']['partial']:.2f}\n")
            f.write(f"  Clear Threshold: {self.default_thresholds['balanced']['clear']:.2f}\n")
            f.write(f"  Confidence Threshold: {self.default_thresholds['balanced']['confidence']:.2f}\n\n")

            f.write("Aggressive Mode (Maximum Speed):\n")
            f.write(f"  Full Blocked Threshold: {self.default_thresholds['aggressive']['full']:.2f}\n")
            f.write(f"  Partial Blocked Threshold: {self.default_thresholds['aggressive']['partial']:.2f}\n")
            f.write(f"  Clear Threshold: {self.default_thresholds['aggressive']['clear']:.2f}\n")
            f.write(f"  Confidence Threshold: {self.default_thresholds['aggressive']['confidence']:.2f}\n")

        print(f"Threshold analysis report saved to: {save_path}")

        return save_path


def run_threshold_analysis(model_path=None, save_results=True):
    """
    Run complete threshold analysis

    Args:
        model_path: Path to trained model
        save_results: Whether to save analysis results

    Returns:
        dict: Analysis results
    """
    import tensorflow as tf
    from utils.dataset import load_dataset

    print("=" * 70)
    print("THRESHOLD ANALYSIS FOR BLIND PATH DETECTION")
    print("=" * 70)

    # Load model
    if model_path is None:
        model_path = MODEL_PATH

    print(f"\nLoading model: {model_path}")
    try:
        model = tf.keras.models.load_model(str(model_path))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load validation data
    print("\nLoading validation dataset...")
    _, val_ds = load_dataset()

    # Create analyzer
    analyzer = ThresholdAnalyzer(model, val_ds)

    # Find optimal threshold
    print("\nFinding optimal threshold...")
    optimal_result = analyzer.find_optimal_thresholds()

    print(f"Optimal threshold: {optimal_result['optimal_threshold']:.3f}")
    print(f"Safety score: {optimal_result['safety_score']:.4f}")
    print(f"Accuracy: {optimal_result['accuracy']:.4f}")

    # Compare safety modes
    print("\nComparing safety modes...")
    mode_comparisons = analyzer.compare_safety_modes()

    for comp in mode_comparisons:
        print(f"\n{comp['mode'].upper()} Mode:")
        print(f"  Accuracy: {comp['accuracy']:.4f}")
        print(f"  Safety Score: {comp['safety_score']:.4f}")
        print(f"  High-risk Errors: {comp['high_risk_errors']}")

    # Save results
    if save_results:
        # Create visualizations
        analyzer.visualize_threshold_analysis()

        # Generate report
        report_path = analyzer.generate_report()

        print(f"\nAnalysis complete!")
        print(f"Visualization saved to: {OUTPUT_DIR}/threshold_analysis.png")
        print(f"Report saved to: {report_path}")

    return {
        "optimal_result": optimal_result,
        "mode_comparisons": mode_comparisons,
        "analyzer": analyzer
    }


if __name__ == "__main__":
    # Run threshold analysis
    results = run_threshold_analysis()

    if results:
        print("\n" + "=" * 70)
        print("THRESHOLD ANALYSIS COMPLETE")
        print("=" * 70)
        print("\nKey Findings:")

        optimal = results["optimal_result"]
        print(f"1. Optimal decision threshold: {optimal['optimal_threshold']:.3f}")
        print(f"2. Maximum safety score: {optimal['safety_score']:.4f}")
        print(f"3. Accuracy at optimal: {optimal['accuracy']:.4f}")

        print("\nSafety Mode Recommendations:")
        for comp in results["mode_comparisons"]:
            print(f"  - {comp['mode'].upper()}: Accuracy={comp['accuracy']:.4f}, "
                  f"Safety={comp['safety_score']:.4f}")