"""
Evaluation Script for Blind Path Detection System
"""

import tensorflow as tf
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

from config import *
from FinalProject.blind_path_detection.utils.dataset import load_dataset
from FinalProject.blind_path_detection.utils.plots import plot_confusion_matrix
from FinalProject.blind_path_detection.analysis.error_analysis import ErrorAnalyzer
from FinalProject.blind_path_detection.explainability.gradcam_utils import GradCAM


class ModelEvaluator:
    """Model evaluator"""

    def __init__(self, model_path=None):
        """
        Initialize evaluator

        Args:
            model_path: Path to trained model
        """
        if model_path is None:
            model_path = MODEL_PATH

        print(f"Loading model: {model_path}")
        self.model = tf.keras.models.load_model(str(model_path))
        self.model_path = Path(model_path)

    def comprehensive_evaluation(self, save_results=True):
        """Execute comprehensive model evaluation"""
        print("Starting comprehensive model evaluation...")

        # Load validation dataset
        _, val_ds = load_dataset()

        # Collect all predictions
        all_predictions = []
        all_labels = []
        all_probs = []

        for batch_images, batch_labels in val_ds:
            batch_probs = self.model.predict(batch_images, verbose=0)
            batch_preds = np.argmax(batch_probs, axis=1)
            batch_true = np.argmax(batch_labels.numpy(), axis=1)

            all_predictions.extend(batch_preds)
            all_labels.extend(batch_true)
            all_probs.extend(batch_probs)

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute metrics
        results = {}

        # 1. Basic metrics
        results['basic_metrics'] = self._compute_basic_metrics(all_labels, all_predictions)

        # 2. Confusion matrix
        results['confusion_matrix'] = confusion_matrix(all_labels, all_predictions)

        # 3. Classification report
        results['classification_report'] = classification_report(
            all_labels, all_predictions,
            target_names=CLASS_NAMES,
            output_dict=True
        )

        # 4. ROC-AUC
        results['roc_auc'] = self._compute_roc_auc(all_labels, all_probs)

        # 5. Error analysis
        print("Performing error analysis...")
        error_analyzer = ErrorAnalyzer(self.model, val_ds)
        error_results = error_analyzer.analyze_errors()
        results['error_analysis'] = error_results

        # 6. Grad-CAM analysis
        print("Performing Grad-CAM analysis...")
        gradcam_results = self._perform_gradcam_analysis(val_ds)
        results['gradcam_analysis'] = gradcam_results

        # 7. Performance benchmarking
        print("Performing performance benchmark...")
        perf_results = self._benchmark_performance()
        results['performance'] = perf_results

        # Save results
        if save_results:
            self._save_evaluation_results(results)

        return results

    def _compute_basic_metrics(self, y_true, y_pred):
        """Compute basic evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
        }

        # Per-class metrics
        for i, class_name in enumerate(CLASS_NAMES):
            mask = y_true == i
            if np.sum(mask) > 0:
                y_true_class = (y_true == i).astype(int)
                y_pred_class = (y_pred == i).astype(int)

                metrics[f'{class_name}_precision'] = precision_score(
                    y_true_class, y_pred_class, zero_division=0
                )
                metrics[f'{class_name}_recall'] = recall_score(
                    y_true_class, y_pred_class, zero_division=0
                )
                metrics[f'{class_name}_f1'] = f1_score(
                    y_true_class, y_pred_class, zero_division=0
                )

        return metrics

    def _compute_roc_auc(self, y_true, y_probs):
        """Compute ROC curves and AUC"""
        from sklearn.preprocessing import label_binarize

        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

        roc_auc = {}
        fpr = {}
        tpr = {}

        for i, class_name in enumerate(CLASS_NAMES):
            fpr[class_name], tpr[class_name], _ = roc_curve(
                y_true_bin[:, i], y_probs[:, i]
            )
            roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])

        # Plot ROC curves
        self._plot_roc_curves(fpr, tpr, roc_auc)

        return roc_auc

    def _plot_roc_curves(self, fpr, tpr, roc_auc):
        """Plot ROC curves"""
        plt.figure(figsize=PLOT_FIGSIZE)

        colors = ['green', 'orange', 'red']
        for i, class_name in enumerate(CLASS_NAMES):
            plt.plot(fpr[class_name], tpr[class_name],
                     color=colors[i], lw=2,
                     label=f'{class_name} (AUC = {roc_auc[class_name]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Blind Path Detection', fontsize=16, pad=20)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        save_path = OUTPUT_DIR / "roc_curves.png"
        plt.savefig(str(save_path), dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        print(f"ROC curves saved: {save_path}")

    def _perform_gradcam_analysis(self, dataset, num_samples=5):
        """Perform Grad-CAM analysis"""
        gradcam = GradCAM(self.model)

        # Collect samples
        sample_images = []
        sample_labels = []

        for batch_images, batch_labels in dataset.take(1):
            sample_images = batch_images[:num_samples]
            sample_labels = batch_labels[:num_samples]
            break

        # Execute Grad-CAM
        results = []

        for i in range(min(num_samples, len(sample_images))):
            image = sample_images[i:i + 1]
            label = np.argmax(sample_labels[i].numpy())

            # Visualize
            save_path = GRADCAM_DIR / f"gradcam_sample_{i}_true_{CLASS_NAMES[label]}.png"
            fig, heatmap, preds = gradcam.visualize(image, save_path=save_path)
            plt.close(fig)

            pred_class = np.argmax(preds)
            results.append({
                'sample_id': i,
                'true_label': CLASS_NAMES[label],
                'pred_label': CLASS_NAMES[pred_class],
                'confidence': np.max(preds),
                'heatmap_path': save_path,
                'correct': label == pred_class
            })

        return results

    def _benchmark_performance(self):
        """Benchmark performance"""
        import time

        # Test inference speed
        dummy_input = np.random.randn(1, *INPUT_SHAPE).astype(np.float32)

        # Warm-up
        for _ in range(10):
            _ = self.model.predict(dummy_input, verbose=0)

        # Batch inference testing
        results = {}

        for batch_size in BENCHMARK_BATCH_SIZES:
            dummy_batch = np.random.randn(batch_size, *INPUT_SHAPE).astype(np.float32)

            # Measure inference time
            start_time = time.time()
            num_runs = max(10, 100 // batch_size)

            for _ in range(num_runs):
                _ = self.model.predict(dummy_batch, verbose=0)

            end_time = time.time()

            total_time = end_time - start_time
            avg_time = total_time / num_runs
            fps = batch_size / avg_time

            results[f'batch_{batch_size}'] = {
                'avg_inference_time': avg_time,
                'fps': fps,
                'throughput': fps * batch_size
            }

        return results

    def _save_evaluation_results(self, results):
        """Save evaluation results"""
        # Save as JSON
        json_path = OUTPUT_DIR / "evaluation_results.json"

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        with open(json_path, 'w') as f:
            json.dump(results, f, default=convert_numpy, indent=2)

        print(f"Evaluation results saved: {json_path}")

        # Generate text report
        self._generate_evaluation_report(results)

    def _generate_evaluation_report(self, results):
        """Generate evaluation report"""
        report_path = OUTPUT_DIR / "evaluation_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Blind Path Detection System - Model Evaluation Report\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. Model Information\n")
            f.write("-" * 80 + "\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Input Shape: {INPUT_SHAPE}\n")
            f.write(f"Output Classes: {CLASS_NAMES}\n\n")

            f.write("2. Basic Performance Metrics\n")
            f.write("-" * 80 + "\n")

            metrics = results['basic_metrics']
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Macro Precision: {metrics['precision_macro']:.4f}\n")
            f.write(f"Macro Recall: {metrics['recall_macro']:.4f}\n")
            f.write(f"Macro F1 Score: {metrics['f1_macro']:.4f}\n\n")

            for class_name in CLASS_NAMES:
                if f'{class_name}_precision' in metrics:
                    f.write(f"{class_name}:\n")
                    f.write(f"  Precision: {metrics[f'{class_name}_precision']:.4f}\n")
                    f.write(f"  Recall: {metrics[f'{class_name}_recall']:.4f}\n")
                    f.write(f"  F1 Score: {metrics[f'{class_name}_f1']:.4f}\n\n")

            f.write("3. Confusion Matrix\n")
            f.write("-" * 80 + "\n")
            cm = results['confusion_matrix']
            f.write("Confusion Matrix:\n")
            for i, row in enumerate(cm):
                f.write(f"{CLASS_NAMES[i]:<20} {str(row)}\n")
            f.write("\n")

            # Calculate per-class recall
            for i, class_name in enumerate(CLASS_NAMES):
                total = cm[i].sum()
                correct = cm[i][i]
                recall = correct / total if total > 0 else 0
                f.write(f"{class_name} Recall: {recall:.4f}\n")
            f.write("\n")

            f.write("4. ROC-AUC Metrics\n")
            f.write("-" * 80 + "\n")
            roc_auc = results['roc_auc']
            for class_name, auc_value in roc_auc.items():
                f.write(f"{class_name}: AUC = {auc_value:.4f}\n")
            f.write("\n")

            f.write("5. Error Analysis\n")
            f.write("-" * 80 + "\n")

            if 'error_analysis' in results:
                error_stats = results['error_analysis']['error_stats']
                risk_analysis = results['error_analysis']['risk_analysis']

                f.write("Error Type Statistics:\n")
                for error_type, count in error_stats.items():
                    f.write(f"  {error_type}: {count} error samples\n")
                f.write("\n")

                f.write("Risk Analysis:\n")
                risk_counts = risk_analysis['risk_counts']
                f.write(f"  High Risk Errors: {risk_counts['high_risk']}\n")
                f.write(f"  Medium Risk Errors: {risk_counts['medium_risk']}\n")
                f.write(f"  Low Risk Errors: {risk_counts['low_risk']}\n\n")

            f.write("6. Performance Benchmark\n")
            f.write("-" * 80 + "\n")

            if 'performance' in results:
                perf_results = results['performance']
                for batch_size, perf in perf_results.items():
                    f.write(f"{batch_size}:\n")
                    f.write(f"  Average Inference Time: {perf['avg_inference_time'] * 1000:.2f} ms\n")
                    f.write(f"  FPS: {perf['fps']:.2f}\n")
                    f.write(f"  Throughput: {perf['throughput']:.2f} samples/second\n\n")

            f.write("7. Conclusions & Recommendations\n")
            f.write("-" * 80 + "\n")

            # Recommendations based on accuracy
            accuracy = metrics['accuracy']
            if accuracy >= 0.95:
                f.write(" Model performance is excellent. Ready for deployment.\n")
            elif accuracy >= 0.90:
                f.write(" Model performance is good. Consider further optimization.\n")
            elif accuracy >= 0.85:
                f.write(" Model performance is moderate. Check data quality and optimize model.\n")
            else:
                f.write(" Model performance needs improvement. Consider retraining or adjusting model architecture.\n")

            # Recommendations based on high-risk errors
            if 'error_analysis' in results:
                high_risk_count = results['error_analysis']['risk_analysis']['risk_counts']['high_risk']
                if high_risk_count > 0:
                    f.write(
                        f"\n Found {high_risk_count} high-risk errors (Blocked misclassified as Clear). Recommendations:\n")
                    f.write("  - Use conservative decision thresholds\n")
                    f.write("  - Increase training data for Blocked classes\n")
                    f.write("  - Implement multi-frame validation mechanism\n")

        print(f"Evaluation report saved: {report_path}")


def main():
    """Main evaluation function"""
    print("Starting model evaluation...")

    # Create evaluator
    evaluator = ModelEvaluator()

    # Execute comprehensive evaluation
    results = evaluator.comprehensive_evaluation(save_results=True)

    # Generate visualizations
    _generate_visualizations(results)

    print("\nEvaluation completed!")
    print(f"All results saved to: {OUTPUT_DIR}")


def _generate_visualizations(results):
    """Generate visualization charts"""
    # Plot confusion matrix
    if 'confusion_matrix' in results:
        cm_path = OUTPUT_DIR / "confusion_matrix.png"
        plot_confusion_matrix(
            results.get('y_true', []),
            results.get('y_pred', []),
            save_path=cm_path
        )

    # Plot error analysis
    if 'error_analysis' in results:
        error_stats = results['error_analysis']['error_stats']
        error_plot_path = OUTPUT_DIR / "error_analysis.png"

        plt.figure(figsize=PLOT_FIGSIZE)
        categories = list(error_stats.keys())
        counts = list(error_stats.values())

        bars = plt.barh(categories, counts, color=['red', 'orange', 'yellow', 'lightcoral', 'lightsalmon', 'gold'])
        for bar, count in zip(bars, counts):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{count}', ha='left', va='center')

        plt.title('Error Type Analysis', fontsize=16, pad=20)
        plt.xlabel('Error Count', fontsize=12)
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(error_plot_path), dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        print(f"Error analysis plot saved: {error_plot_path}")


# Add these to your evaluate.py or create a separate analysis script

def run_complete_analysis():
    """Run all analysis modules"""
    print("Running complete system analysis...")

    # 1. Threshold analysis
    from analysis.threshold_analysis import run_threshold_analysis
    print("\n1. Running threshold analysis...")
    threshold_results = run_threshold_analysis()

    # 2. Speed benchmark
    from analysis.speed_benchmark import run_complete_benchmark
    print("\n2. Running speed benchmark...")
    benchmark_results = run_complete_benchmark()

    # 3. Error analysis (if you have it)
    print("\n3. Running error analysis...")
    # ... error analysis code ...

    print("\nAnalysis complete!")
    return {
        "threshold": threshold_results,
        "benchmark": benchmark_results
    }

if __name__ == "__main__":
    main()