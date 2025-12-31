"""
Error Analysis for Blind Path Detection System
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from config import *


class ErrorAnalyzer:
    """Error analysis utility class"""

    def __init__(self, model, dataset):
        """
        Initialize error analyzer

        Args:
            model: Trained model
            dataset: Validation dataset
        """
        self.model = model
        self.dataset = dataset

    def analyze_errors(self):
        """Execute complete error analysis"""
        print("Starting error analysis...")

        # Collect predictions and true labels
        y_true = []
        y_pred = []
        all_probs = []
        error_samples = []

        for batch_images, batch_labels in self.dataset:
            batch_preds = self.model.predict(batch_images, verbose=0)
            batch_y_pred = np.argmax(batch_preds, axis=1)
            batch_y_true = np.argmax(batch_labels.numpy(), axis=1)

            y_true.extend(batch_y_true)
            y_pred.extend(batch_y_pred)
            all_probs.extend(batch_preds)

            # Collect error samples
            for i in range(len(batch_y_true)):
                if batch_y_true[i] != batch_y_pred[i]:
                    error_samples.append({
                        'image': batch_images[i].numpy(),
                        'true_label': batch_y_true[i],
                        'pred_label': batch_y_pred[i],
                        'true_class': CLASS_NAMES[batch_y_true[i]],
                        'pred_class': CLASS_NAMES[batch_y_pred[i]],
                        'probabilities': batch_preds[i]
                    })

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        all_probs = np.array(all_probs)

        # Compute evaluation metrics
        report = classification_report(
            y_true, y_pred,
            target_names=CLASS_NAMES,
            output_dict=True
        )

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Categorize errors
        error_stats = self._categorize_errors(y_true, y_pred)

        # Risk analysis
        risk_analysis = self._analyze_risk(y_true, y_pred, all_probs)

        return {
            'report': report,
            'confusion_matrix': cm,
            'error_stats': error_stats,
            'risk_analysis': risk_analysis,
            'error_samples': error_samples,
            'y_true': y_true,
            'y_pred': y_pred,
            'probs': all_probs
        }

    def _categorize_errors(self, y_true, y_pred):
        """Categorize error types"""
        error_categories = {
            'Clear_as_Blocked': 0,  # Clear misclassified as Blocked
            'Partial_as_Full': 0,  # Partial misclassified as Full
            'Full_as_Partial': 0,  # Full misclassified as Partial
            'Clear_as_Partial': 0,  # Clear misclassified as Partial
            'Clear_as_Full': 0,  # Clear misclassified as Full
            'Blocked_as_Clear': 0  # Any Blocked misclassified as Clear
        }

        for true, pred in zip(y_true, y_pred):
            if true == 0 and pred in [1, 2]:  # Clear -> Blocked
                error_categories['Clear_as_Blocked'] += 1
                if pred == 1:
                    error_categories['Clear_as_Partial'] += 1
                else:
                    error_categories['Clear_as_Full'] += 1
            elif true == 1 and pred == 2:  # Partial -> Full
                error_categories['Partial_as_Full'] += 1
            elif true == 2 and pred == 1:  # Full -> Partial
                error_categories['Full_as_Partial'] += 1
            elif true in [1, 2] and pred == 0:  # Blocked -> Clear
                error_categories['Blocked_as_Clear'] += 1

        return error_categories

    def _analyze_risk(self, y_true, y_pred, probs):
        """Analyze risk levels"""
        risk_cases = {
            'high_risk': 0,  # High risk: Blocked misclassified as Clear
            'medium_risk': 0,  # Medium risk: Severity misclassification
            'low_risk': 0  # Low risk: Minor misclassification
        }

        risk_details = []

        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if true != pred:
                # High risk: Blocked misclassified as Clear
                if true in [1, 2] and pred == 0:
                    risk_level = 'high'
                    risk_cases['high_risk'] += 1
                    description = f"Blocked misclassified as Clear: {CLASS_NAMES[true]} -> {CLASS_NAMES[pred]}"

                # Medium risk: Full and Partial misclassified
                elif (true == 1 and pred == 2) or (true == 2 and pred == 1):
                    risk_level = 'medium'
                    risk_cases['medium_risk'] += 1
                    description = f"Severity misclassification: {CLASS_NAMES[true]} -> {CLASS_NAMES[pred]}"

                # Low risk: Clear misclassified as Blocked
                else:
                    risk_level = 'low'
                    risk_cases['low_risk'] += 1
                    description = f"Safe misclassification: {CLASS_NAMES[true]} -> {CLASS_NAMES[pred]}"

                risk_details.append({
                    'index': i,
                    'true_label': true,
                    'pred_label': pred,
                    'true_class': CLASS_NAMES[true],
                    'pred_class': CLASS_NAMES[pred],
                    'probabilities': probs[i].tolist(),
                    'risk_level': risk_level,
                    'description': description
                })

        return {
            'risk_counts': risk_cases,
            'risk_details': risk_details
        }

    def save_error_samples(self, error_samples, save_dir=None):
        """Save error sample images"""
        if save_dir is None:
            save_dir = ERROR_SAMPLES_DIR

        save_dir = Path(save_dir)

        # Create subdirectories
        categories = ['Clear_as_Blocked', 'Partial_as_Full', 'Full_as_Partial', 'Blocked_as_Clear']
        for cat in categories:
            (save_dir / cat).mkdir(parents=True, exist_ok=True)

        saved_count = 0

        for sample in error_samples:
            # Determine error category
            if sample['true_label'] == 0 and sample['pred_label'] in [1, 2]:
                category = 'Clear_as_Blocked'
            elif sample['true_label'] == 1 and sample['pred_label'] == 2:
                category = 'Partial_as_Full'
            elif sample['true_label'] == 2 and sample['pred_label'] == 1:
                category = 'Full_as_Partial'
            elif sample['true_label'] in [1, 2] and sample['pred_label'] == 0:
                category = 'Blocked_as_Clear'
            else:
                continue

            # Convert image format and save
            img = sample['image']
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

            filename = f"{sample['true_class']}_as_{sample['pred_class']}_{saved_count}.jpg"
            save_path = save_dir / category / filename

            # Convert channel order if RGB format
            if len(img.shape) == 3 and img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(save_path), img)
            saved_count += 1

        print(f"Saved {saved_count} error samples to {save_dir}")

    def generate_report(self, analysis_results, save_path=None):
        """Generate error analysis report"""
        if save_path is None:
            save_path = OUTPUT_DIR / "error_analysis_report.txt"

        report = analysis_results['report']
        error_stats = analysis_results['error_stats']
        risk_analysis = analysis_results['risk_analysis']

        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Blind Path Detection System - Error Analysis Report\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. Classification Performance Report\n")
            f.write("-" * 80 + "\n")

            for class_name in CLASS_NAMES:
                class_report = report[class_name]
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {class_report['precision']:.4f}\n")
                f.write(f"  Recall: {class_report['recall']:.4f}\n")
                f.write(f"  F1 Score: {class_report['f1-score']:.4f}\n")
                f.write(f"  Support: {class_report['support']}\n")

            f.write(f"\nOverall Accuracy: {report['accuracy']:.4f}\n")
            f.write(f"Macro Average F1: {report['macro avg']['f1-score']:.4f}\n")
            f.write(f"Weighted Average F1: {report['weighted avg']['f1-score']:.4f}\n\n")

            f.write("2. Error Type Analysis\n")
            f.write("-" * 80 + "\n")

            for error_type, count in error_stats.items():
                f.write(f"{error_type}: {count} error samples\n")

            total_errors = sum(error_stats.values())
            if total_errors > 0:
                f.write(f"\nError Type Percentages:\n")
                for error_type, count in error_stats.items():
                    percentage = (count / total_errors) * 100
                    f.write(f"  {error_type}: {percentage:.1f}%\n")

            f.write("\n3. Risk Analysis\n")
            f.write("-" * 80 + "\n")

            risk_counts = risk_analysis['risk_counts']
            f.write(f"High Risk Errors (Blocked misclassified as Clear): {risk_counts['high_risk']}\n")
            f.write(f"Medium Risk Errors (Severity misclassification): {risk_counts['medium_risk']}\n")
            f.write(f"Low Risk Errors (Safe misclassification): {risk_counts['low_risk']}\n")

            if total_errors > 0:
                high_risk_percentage = (risk_counts['high_risk'] / total_errors) * 100
                f.write(f"\nHigh Risk Error Percentage: {high_risk_percentage:.1f}%\n")

                if high_risk_percentage > 10:
                    f.write("\n⚠️ Warning: High risk error percentage is high. Recommendations:\n")
                    f.write("  - Increase training data for Blocked classes\n")
                    f.write("  - Adjust decision thresholds\n")
                    f.write("  - Use more conservative decision mode\n")

            f.write("\n4. Recommended Improvements\n")
            f.write("-" * 80 + "\n")

            f.write("a. Data Level:\n")
            f.write("   - Increase sample diversity for Partial Blocked and Full Blocked\n")
            f.write("   - Balance sample counts across classes\n")
            f.write("   - Add more edge cases\n\n")

            f.write("b. Model Level:\n")
            f.write("   - Use deeper network architectures\n")
            f.write("   - Try different loss functions (e.g., Focal Loss)\n")
            f.write("   - Adjust class weights\n\n")

            f.write("c. Decision Level:\n")
            f.write("   - Use conservative thresholds to reduce high-risk errors\n")
            f.write("   - Implement confidence filtering\n")
            f.write("   - Add multi-frame smoothing\n")

        print(f"Error analysis report saved: {save_path}")
        return save_path