"""
Speed Benchmark for Blind Path Detection System

This module benchmarks the inference speed and performance of different models.
It provides detailed timing analysis for deployment decisions.
"""

import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import *
from models.model_factory import build_model


class SpeedBenchmark:
    """Benchmarks model inference speed and performance"""

    def __init__(self, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
        """
        Initialize speed benchmark

        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Benchmark configurations
        self.batch_sizes = [1, 2, 4, 8, 16, 32]
        self.num_runs = 100
        self.warmup_runs = 10

        # Model configurations to test
        self.model_configs = [
            {
                "name": "CNN_V1",
                "type": "cnn_v1",
                "transfer": False,
                "description": "Deep CNN (Baseline)"
            },
            {
                "name": "CNN_V2",
                "type": "cnn_v2",
                "transfer": False,
                "description": "Light CNN (Fast)"
            },
            {
                "name": "MobileNet (No Transfer)",
                "type": "mobilenet",
                "transfer": False,
                "description": "MobileNet without pretraining"
            },
            {
                "name": "MobileNet (Transfer)",
                "type": "mobilenet",
                "transfer": True,
                "description": "MobileNet with ImageNet weights"
            }
        ]

    def benchmark_single_model(self, model_config):
        """
        Benchmark a single model configuration

        Args:
            model_config: Model configuration dictionary

        Returns:
            dict: Benchmark results
        """
        print(f"Benchmarking {model_config['name']}...")

        # Build model
        model = build_model(
            model_type=model_config["type"],
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            use_transfer=model_config["transfer"]
        )

        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        results = {
            "model_name": model_config["name"],
            "model_type": model_config["type"],
            "transfer_learning": model_config["transfer"],
            "description": model_config["description"],
            "batch_results": {}
        }

        # Benchmark for each batch size
        for batch_size in self.batch_sizes:
            print(f"  Batch size: {batch_size}...", end=" ", flush=True)

            # Create dummy input
            dummy_input = np.random.randn(batch_size, *self.input_shape).astype(np.float32)

            # Warm-up runs
            for _ in range(self.warmup_runs):
                _ = model.predict(dummy_input, verbose=0)

            # Timing runs
            start_time = time.time()
            for _ in range(self.num_runs):
                _ = model.predict(dummy_input, verbose=0)
            end_time = time.time()

            # Calculate metrics
            total_time = end_time - start_time
            avg_time_per_batch = total_time / self.num_runs
            avg_time_per_image = avg_time_per_batch / batch_size
            fps = batch_size / avg_time_per_batch
            throughput = fps * batch_size

            batch_result = {
                "batch_size": batch_size,
                "total_time": total_time,
                "avg_time_per_batch": avg_time_per_batch,
                "avg_time_per_image": avg_time_per_image,
                "fps": fps,
                "throughput": throughput,
                "num_runs": self.num_runs
            }

            results["batch_results"][batch_size] = batch_result
            print(f"Done. FPS: {fps:.1f}")

        # Calculate overall metrics
        results["overall_metrics"] = self._calculate_overall_metrics(results["batch_results"])

        return results

    def _calculate_overall_metrics(self, batch_results):
        """Calculate overall metrics from batch results"""
        # Find best batch size for throughput
        throughputs = [r["throughput"] for r in batch_results.values()]
        best_batch_idx = np.argmax(throughputs)
        best_batch_size = self.batch_sizes[best_batch_idx]
        best_throughput = throughputs[best_batch_idx]

        # Calculate latency for single image
        single_image_latency = batch_results[1]["avg_time_per_image"] * 1000  # Convert to ms

        return {
            "best_batch_size": best_batch_size,
            "max_throughput": best_throughput,
            "single_image_latency_ms": single_image_latency,
            "realtime_fps_estimate": min(30, batch_results[1]["fps"])  # Cap at 30 for real-time
        }

    def benchmark_all_models(self):
        """Benchmark all model configurations"""
        print("=" * 70)
        print("SPEED BENCHMARK - BLIND PATH DETECTION SYSTEM")
        print("=" * 70)

        all_results = []

        for config in self.model_configs:
            try:
                results = self.benchmark_single_model(config)
                all_results.append(results)
                print()
            except Exception as e:
                print(f"\nError benchmarking {config['name']}: {e}")

        return all_results

    def analyze_memory_usage(self, model):
        """
        Analyze memory usage of a model

        Args:
            model: Keras model

        Returns:
            dict: Memory usage statistics
        """
        # Calculate number of parameters
        total_params = model.count_params()
        trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params

        # Estimate memory usage (rough estimate)
        # 4 bytes per parameter for float32
        memory_mb = total_params * 4 / (1024 * 1024)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
            "estimated_memory_mb": memory_mb,
            "model_size_mb": memory_mb  # Same as estimated memory for simplicity
        }

    def create_comparison_report(self, benchmark_results, save_path=None):
        """
        Create comparison report of all models

        Args:
            benchmark_results: List of benchmark results
            save_path: Path to save report

        Returns:
            str: Report content
        """
        if save_path is None:
            save_path = OUTPUT_DIR / "speed_benchmark_report.txt"

        # Create report
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("SPEED BENCHMARK REPORT - BLIND PATH DETECTION SYSTEM")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Input Shape: {self.input_shape}")
        report_lines.append(f"Number of Classes: {self.num_classes}")
        report_lines.append("")

        # Summary table
        report_lines.append("1. EXECUTIVE SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Model':<25} {'Latency (ms)':<15} {'Max FPS':<15} {'Best Batch':<15} {'Params (M)':<15}")
        report_lines.append("-" * 80)

        for result in benchmark_results:
            model_name = result["model_name"]
            overall = result["overall_metrics"]
            memory_info = self.analyze_memory_usage(
                build_model(
                    model_type=result["model_type"],
                    input_shape=self.input_shape,
                    num_classes=self.num_classes,
                    use_transfer=result["transfer_learning"]
                )
            )

            latency = overall["single_image_latency_ms"]
            max_fps = overall["max_throughput"]
            best_batch = overall["best_batch_size"]
            params_m = memory_info["total_parameters"] / 1_000_000

            report_lines.append(f"{model_name:<25} {latency:<15.1f} {max_fps:<15.1f} {best_batch:<15} {params_m:<15.2f}")

        report_lines.append("")

        # Detailed analysis for each model
        report_lines.append("2. DETAILED ANALYSIS BY MODEL")
        report_lines.append("-" * 80)

        for result in benchmark_results:
            model_name = result["model_name"]
            description = result["description"]
            overall = result["overall_metrics"]
            memory_info = self.analyze_memory_usage(
                build_model(
                    model_type=result["model_type"],
                    input_shape=self.input_shape,
                    num_classes=self.num_classes,
                    use_transfer=result["transfer_learning"]
                )
            )

            report_lines.append(f"\n{model_name}")
            report_lines.append(f"Description: {description}")
            report_lines.append(f"Transfer Learning: {result['transfer_learning']}")
            report_lines.append(f"Model Type: {result['model_type']}")
            report_lines.append("")

            report_lines.append("Performance Metrics:")
            report_lines.append(f"  Single Image Latency: {overall['single_image_latency_ms']:.1f} ms")
            report_lines.append(f"  Maximum Throughput: {overall['max_throughput']:.1f} images/sec")
            report_lines.append(f"  Real-time FPS Estimate: {overall['realtime_fps_estimate']:.1f} FPS")
            report_lines.append(f"  Optimal Batch Size: {overall['best_batch_size']}")
            report_lines.append("")

            report_lines.append("Memory Usage:")
            report_lines.append(f"  Total Parameters: {memory_info['total_parameters']:,}")
            report_lines.append(f"  Trainable Parameters: {memory_info['trainable_parameters']:,}")
            report_lines.append(f"  Non-trainable Parameters: {memory_info['non_trainable_parameters']:,}")
            report_lines.append(f"  Estimated Memory: {memory_info['estimated_memory_mb']:.2f} MB")
            report_lines.append("")

            report_lines.append("Batch Size Performance:")
            report_lines.append(f"{'Batch Size':<12} {'Latency/Img (ms)':<20} {'FPS':<15} {'Throughput':<15}")
            report_lines.append("-" * 60)

            for batch_size, batch_result in result["batch_results"].items():
                latency_ms = batch_result["avg_time_per_image"] * 1000
                fps = batch_result["fps"]
                throughput = batch_result["throughput"]

                report_lines.append(f"{batch_size:<12} {latency_ms:<20.2f} {fps:<15.1f} {throughput:<15.1f}")

            report_lines.append("")

        # Recommendations
        report_lines.append("3. DEPLOYMENT RECOMMENDATIONS")
        report_lines.append("-" * 80)

        # Find best model for each use case
        models_by_latency = sorted(benchmark_results,
                                 key=lambda x: x["overall_metrics"]["single_image_latency_ms"])
        models_by_throughput = sorted(benchmark_results,
                                    key=lambda x: x["overall_metrics"]["max_throughput"],
                                    reverse=True)
        models_by_memory = sorted(benchmark_results,
                                key=lambda x: self.analyze_memory_usage(
                                    build_model(
                                        model_type=x["model_type"],
                                        input_shape=self.input_shape,
                                        num_classes=self.num_classes,
                                        use_transfer=x["transfer_learning"]
                                    )
                                )["estimated_memory_mb"])

        report_lines.append("\nA. For Real-time Applications (Low Latency):")
        best_latency = models_by_latency[0]
        report_lines.append(f"   Recommended: {best_latency['model_name']}")
        report_lines.append(f"   Latency: {best_latency['overall_metrics']['single_image_latency_ms']:.1f} ms")
        report_lines.append(f"   Estimated FPS: {best_latency['overall_metrics']['realtime_fps_estimate']:.1f}")

        report_lines.append("\nB. For Batch Processing (High Throughput):")
        best_throughput = models_by_throughput[0]
        report_lines.append(f"   Recommended: {best_throughput['model_name']}")
        report_lines.append(f"   Throughput: {best_throughput['overall_metrics']['max_throughput']:.1f} images/sec")
        report_lines.append(f"   Optimal Batch Size: {best_throughput['overall_metrics']['best_batch_size']}")

        report_lines.append("\nC. For Resource-constrained Devices (Low Memory):")
        best_memory = models_by_memory[0]
        memory_info = self.analyze_memory_usage(
            build_model(
                model_type=best_memory["model_type"],
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                use_transfer=best_memory["transfer_learning"]
            )
        )
        report_lines.append(f"   Recommended: {best_memory['model_name']}")
        report_lines.append(f"   Memory Usage: {memory_info['estimated_memory_mb']:.2f} MB")
        report_lines.append(f"   Parameters: {memory_info['total_parameters']:,}")

        report_lines.append("\nD. For Maximum Accuracy (Typically):")
        report_lines.append("   Recommended: MobileNet (Transfer)")
        report_lines.append("   Reason: Transfer learning provides better feature extraction")
        report_lines.append("   Note: Verify accuracy with your specific dataset")

        report_lines.append("\n4. PERFORMANCE VS ACCURACY TRADE-OFF")
        report_lines.append("-" * 80)
        report_lines.append("\nNote: These are general guidelines. Actual accuracy depends on your dataset.")
        report_lines.append("\nModel Type               | Speed     | Accuracy  | Use Case")
        report_lines.append("-" * 70)
        report_lines.append("CNN_V1 (Deep)           | Medium    | High      | General purpose, good balance")
        report_lines.append("CNN_V2 (Light)          | Fast      | Medium    | Real-time, resource-constrained")
        report_lines.append("MobileNet (No Transfer) | Fast      | Low-Medium| Fast inference, limited training data")
        report_lines.append("MobileNet (Transfer)    | Medium    | High      | Maximum accuracy, sufficient data")

        report_lines.append("\n5. OPTIMIZATION SUGGESTIONS")
        report_lines.append("-" * 80)
        report_lines.append("\n1. For Faster Inference:")
        report_lines.append("   - Use CNN_V2 or MobileNet architectures")
        report_lines.append("   - Reduce input image size (e.g., 160x160 instead of 224x224)")
        report_lines.append("   - Use batch size 8-16 for optimal throughput")
        report_lines.append("   - Enable GPU acceleration if available")

        report_lines.append("\n2. For Lower Memory Usage:")
        report_lines.append("   - Use CNN_V2 (lightest)")
        report_lines.append("   - Consider model quantization")
        report_lines.append("   - Use fp16 precision if supported")

        report_lines.append("\n3. For Better Accuracy:")
        report_lines.append("   - Use MobileNet with transfer learning")
        report_lines.append("   - Ensure sufficient training data")
        report_lines.append("   - Use proper data augmentation")
        report_lines.append("   - Fine-tune hyperparameters")

        # Write report to file
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"Benchmark report saved to: {save_path}")

        return '\n'.join(report_lines)

    def visualize_benchmark_results(self, benchmark_results, save_path=None):
        """
        Create visualizations of benchmark results

        Args:
            benchmark_results: List of benchmark results
            save_path: Base path for saving visualizations
        """
        if save_path is None:
            save_path = OUTPUT_DIR

        save_path = Path(save_path)

        # Create multiple visualizations
        self._plot_latency_comparison(benchmark_results, save_path / "latency_comparison.png")
        self._plot_throughput_comparison(benchmark_results, save_path / "throughput_comparison.png")
        self._plot_batch_size_analysis(benchmark_results, save_path / "batch_size_analysis.png")
        self._plot_tradeoff_analysis(benchmark_results, save_path / "tradeoff_analysis.png")

    def _plot_latency_comparison(self, benchmark_results, save_path):
        """Plot latency comparison across models"""
        plt.figure(figsize=(12, 8))

        model_names = []
        latencies = []

        for result in benchmark_results:
            model_names.append(result["model_name"])
            latencies.append(result["overall_metrics"]["single_image_latency_ms"])

        # Create bar chart
        bars = plt.bar(model_names, latencies, color=['blue', 'green', 'orange', 'red'])

        # Add value labels
        for bar, latency in zip(bars, latencies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{latency:.1f} ms', ha='center', va='bottom', fontsize=10)

        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Latency per Image (ms)', fontsize=12)
        plt.title('Model Latency Comparison', fontsize=14, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Latency comparison plot saved to: {save_path}")

    def _plot_throughput_comparison(self, benchmark_results, save_path):
        """Plot throughput comparison across models"""
        plt.figure(figsize=(12, 8))

        model_names = []
        throughputs = []

        for result in benchmark_results:
            model_names.append(result["model_name"])
            throughputs.append(result["overall_metrics"]["max_throughput"])

        # Create bar chart
        bars = plt.bar(model_names, throughputs, color=['blue', 'green', 'orange', 'red'])

        # Add value labels
        for bar, throughput in zip(bars, throughputs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{throughput:.0f} img/s', ha='center', va='bottom', fontsize=10)

        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Maximum Throughput (images/second)', fontsize=12)
        plt.title('Model Throughput Comparison', fontsize=14, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Throughput comparison plot saved to: {save_path}")

    def _plot_batch_size_analysis(self, benchmark_results, save_path):
        """Plot batch size analysis for each model"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, result in enumerate(benchmark_results):
            if idx >= len(axes):
                break

            ax = axes[idx]
            model_name = result["model_name"]
            batch_results = result["batch_results"]

            batch_sizes = []
            latencies = []
            throughputs = []

            for batch_size, batch_result in batch_results.items():
                batch_sizes.append(batch_size)
                latencies.append(batch_result["avg_time_per_image"] * 1000)
                throughputs.append(batch_result["throughput"])

            # Plot latency
            color = 'tab:red'
            ax.set_xlabel('Batch Size', fontsize=10)
            ax.set_ylabel('Latency per Image (ms)', color=color, fontsize=10)
            ax.plot(batch_sizes, latencies, 'o-', color=color, linewidth=2)
            ax.tick_params(axis='y', labelcolor=color)
            ax.grid(True, alpha=0.3)

            # Plot throughput on secondary axis
            ax2 = ax.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Throughput (img/s)', color=color, fontsize=10)
            ax2.plot(batch_sizes, throughputs, 's--', color=color, linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color)

            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')

        plt.suptitle('Batch Size Analysis for Different Models', fontsize=14, fontweight='bold')
        plt.tight_layout()

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Batch size analysis plot saved to: {save_path}")

    def _plot_tradeoff_analysis(self, benchmark_results, save_path):
        """Plot accuracy vs speed tradeoff analysis"""
        plt.figure(figsize=(10, 8))

        # Note: These accuracy values are estimates
        # In practice, you should use actual accuracy from your evaluation
        accuracy_estimates = {
            "CNN_V1": 0.88,
            "CNN_V2": 0.85,
            "MobileNet (No Transfer)": 0.82,
            "MobileNet (Transfer)": 0.92
        }

        model_names = []
        latencies = []
        accuracies = []

        for result in benchmark_results:
            model_name = result["model_name"]
            if model_name in accuracy_estimates:
                model_names.append(model_name)
                latencies.append(result["overall_metrics"]["single_image_latency_ms"])
                accuracies.append(accuracy_estimates[model_name] * 100)  # Convert to percentage

        # Create scatter plot
        scatter = plt.scatter(latencies, accuracies, s=200, alpha=0.7)

        # Add model labels
        for i, (model, lat, acc) in enumerate(zip(model_names, latencies, accuracies)):
            plt.annotate(model, (lat, acc), fontsize=10,
                        xytext=(10, 5), textcoords='offset points')

        plt.xlabel('Latency per Image (ms)', fontsize=12)
        plt.ylabel('Estimated Accuracy (%)', fontsize=12)
        plt.title('Accuracy vs Speed Trade-off Analysis', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Add quadrants
        avg_latency = np.mean(latencies)
        avg_accuracy = np.mean(accuracies)

        plt.axhline(y=avg_accuracy, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=avg_latency, color='gray', linestyle='--', alpha=0.5)

        # Add quadrant labels
        plt.text(avg_latency/2, avg_accuracy*1.1, 'Fast & Accurate\n(Ideal)',
                ha='center', fontsize=10, fontweight='bold', color='green')
        plt.text(avg_latency*1.5, avg_accuracy*1.1, 'Accurate but Slow',
                ha='center', fontsize=10, fontweight='bold', color='blue')
        plt.text(avg_latency/2, avg_accuracy*0.9, 'Fast but Less Accurate',
                ha='center', fontsize=10, fontweight='bold', color='orange')
        plt.text(avg_latency*1.5, avg_accuracy*0.9, 'Slow & Less Accurate\n(Avoid)',
                ha='center', fontsize=10, fontweight='bold', color='red')

        plt.tight_layout()

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Trade-off analysis plot saved to: {save_path}")


def run_complete_benchmark():
    """Run complete speed benchmark analysis"""
    print("=" * 70)
    print("COMPLETE SPEED BENCHMARK ANALYSIS")
    print("=" * 70)

    # Create benchmark instance
    benchmark = SpeedBenchmark()

    # Run benchmarks for all models
    print("\nRunning benchmarks for all model configurations...")
    benchmark_results = benchmark.benchmark_all_models()

    if not benchmark_results:
        print("No benchmark results available. Exiting.")
        return

    # Create visualizations
    print("\nCreating visualizations...")
    benchmark.visualize_benchmark_results(benchmark_results)

    # Generate report
    print("\nGenerating benchmark report...")
    report = benchmark.create_comparison_report(benchmark_results)

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print("\nModel Performance Ranking (by Latency):")
    print("-" * 50)

    # Sort by latency
    sorted_results = sorted(benchmark_results,
                          key=lambda x: x["overall_metrics"]["single_image_latency_ms"])

    for i, result in enumerate(sorted_results, 1):
        model_name = result["model_name"]
        latency = result["overall_metrics"]["single_image_latency_ms"]
        throughput = result["overall_metrics"]["max_throughput"]

        print(f"{i}. {model_name}")
        print(f"   Latency: {latency:.1f} ms per image")
        print(f"   Throughput: {throughput:.0f} images/second")
        print()

    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 50)

    fastest = sorted_results[0]
    highest_throughput = max(benchmark_results,
                           key=lambda x: x["overall_metrics"]["max_throughput"])

    print(f"1. For Real-time Applications:")
    print(f"   → {fastest['model_name']}")
    print(f"   Latency: {fastest['overall_metrics']['single_image_latency_ms']:.1f} ms")
    print(f"   Estimated FPS: {fastest['overall_metrics']['realtime_fps_estimate']:.1f}")

    print(f"\n2. For Batch Processing:")
    print(f"   → {highest_throughput['model_name']}")
    print(f"   Throughput: {highest_throughput['overall_metrics']['max_throughput']:.0f} img/s")
    print(f"   Optimal Batch Size: {highest_throughput['overall_metrics']['best_batch_size']}")

    print(f"\n3. For General Purpose (Balance):")
    # Find model closest to median latency
    latencies = [r["overall_metrics"]["single_image_latency_ms"] for r in benchmark_results]
    median_idx = np.argsort(latencies)[len(latencies)//2]
    balanced_model = benchmark_results[median_idx]

    print(f"   → {balanced_model['model_name']}")
    print(f"   Latency: {balanced_model['overall_metrics']['single_image_latency_ms']:.1f} ms")
    print(f"   Throughput: {balanced_model['overall_metrics']['max_throughput']:.0f} img/s")

    print(f"\nOutput saved to:")
    print(f"  - {OUTPUT_DIR}/speed_benchmark_report.txt")
    print(f"  - {OUTPUT_DIR}/latency_comparison.png")
    print(f"  - {OUTPUT_DIR}/throughput_comparison.png")
    print(f"  - {OUTPUT_DIR}/batch_size_analysis.png")
    print(f"  - {OUTPUT_DIR}/tradeoff_analysis.png")

    return benchmark_results


def benchmark_pretrained_model(model_path):
    """
    Benchmark a specific pretrained model

    Args:
        model_path: Path to pretrained model file

    Returns:
        dict: Benchmark results
    """
    import tensorflow as tf

    print(f"\nBenchmarking pretrained model: {model_path}")

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Create benchmark instance
    benchmark = SpeedBenchmark()

    # Create model config for reporting
    model_config = {
        "name": "Pretrained Model",
        "type": "custom",
        "transfer": True,
        "description": f"Loaded from {model_path}"
    }

    # Benchmark the model
    results = benchmark.benchmark_single_model(model_config)

    # Add memory analysis
    memory_info = benchmark.analyze_memory_usage(model)
    results["memory_info"] = memory_info

    # Print results
    print("\nBenchmark Results:")
    print("-" * 40)
    print(f"Model: {model_path}")
    print(f"Single Image Latency: {results['overall_metrics']['single_image_latency_ms']:.1f} ms")
    print(f"Maximum Throughput: {results['overall_metrics']['max_throughput']:.0f} images/second")
    print(f"Real-time FPS Estimate: {results['overall_metrics']['realtime_fps_estimate']:.1f}")
    print(f"Optimal Batch Size: {results['overall_metrics']['best_batch_size']}")
    print(f"\nMemory Usage:")
    print(f"  Total Parameters: {memory_info['total_parameters']:,}")
    print(f"  Estimated Memory: {memory_info['estimated_memory_mb']:.2f} MB")

    return results


if __name__ == "__main__":
    # Run complete benchmark
    results = run_complete_benchmark()

    # If a model file is provided as argument, also benchmark it
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if Path(model_path).exists():
            benchmark_pretrained_model(model_path)