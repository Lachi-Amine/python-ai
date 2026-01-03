#!/usr/bin/env python3
"""
Evaluation Results Graph Generator
Creates comprehensive visualizations from evaluation_results.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from pathlib import Path

# Set style for better looking graphs
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_evaluation_data(file_path):
    """Load evaluation data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def plot_confusion_matrix(data, save_path=None):
    """Plot confusion matrix heatmap"""
    cm = np.array(data['confusion_matrix'])
    class_names = ['Clear', 'Left Blocked', 'Right Blocked', 'Fully Blocked']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Add text annotations for percentages
    total = cm.sum()
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'{percentage:.1f}%', 
                    ha='center', va='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_metrics(data, save_path=None):
    """Plot precision, recall, and F1-score for each class"""
    classes = ['Clear', 'Left Blocked', 'Right Blocked', 'Fully Blocked']
    
    precision = [data['basic_metrics']['Clear_precision'],
                data['basic_metrics']['Left Blocked_precision'],
                data['basic_metrics']['Right Blocked_precision'],
                data['basic_metrics']['Fully Blocked_precision']]
    
    recall = [data['basic_metrics']['Clear_recall'],
             data['basic_metrics']['Left Blocked_recall'],
             data['basic_metrics']['Right Blocked_recall'],
             data['basic_metrics']['Fully Blocked_recall']]
    
    f1 = [data['basic_metrics']['Clear_f1'],
          data['basic_metrics']['Left Blocked_f1'],
          data['basic_metrics']['Right Blocked_f1'],
          data['basic_metrics']['Fully Blocked_f1']]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_prediction_distribution(data, save_path=None):
    """Plot distribution of predictions vs true labels"""
    y_true = np.array(data['y_true'])
    y_pred = np.array(data['y_pred'])
    class_names = ['Clear', 'Left Blocked', 'Right Blocked', 'Fully Blocked']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # True labels distribution
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    ax1.bar([class_names[i-1] for i in unique_true], counts_true, color='skyblue', alpha=0.7)
    ax1.set_title('True Labels Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(counts_true):
        ax1.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Predicted labels distribution
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    ax2.bar([class_names[i-1] for i in unique_pred], counts_pred, color='lightcoral', alpha=0.7)
    ax2.set_title('Predicted Labels Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(counts_pred):
        ax2.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_overall_metrics(data, save_path=None):
    """Plot overall performance metrics"""
    metrics = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1-Score\n(Macro)']
    values = [
        data['basic_metrics']['accuracy'],
        data['basic_metrics']['precision_macro'],
        data['basic_metrics']['recall_macro'],
        data['basic_metrics']['f1_macro']
    ]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, values, color=colors, alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_accuracy(data, save_path=None):
    """Plot accuracy for each class"""
    classes = ['Clear', 'Left Blocked', 'Right Blocked', 'Fully Blocked']
    cm = np.array(data['confusion_matrix'])
    
    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(len(classes)):
        class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        class_accuracies.append(class_accuracy)
    
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, class_accuracies, color=colors, alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(data, save_path=None):
    """Create a summary report with key metrics"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create summary text
    summary_text = f"""
EVALUATION SUMMARY REPORT
{'='*50}

Overall Performance:
• Accuracy: {data['basic_metrics']['accuracy']:.3f} ({data['basic_metrics']['accuracy']*100:.1f}%)
• Macro Precision: {data['basic_metrics']['precision_macro']:.3f}
• Macro Recall: {data['basic_metrics']['recall_macro']:.3f}
• Macro F1-Score: {data['basic_metrics']['f1_macro']:.3f}

Class-wise Performance:
• Clear: Precision={data['basic_metrics']['Clear_precision']:.3f}, Recall={data['basic_metrics']['Clear_recall']:.3f}, F1={data['basic_metrics']['Clear_f1']:.3f}
• Left Blocked: Precision={data['basic_metrics']['Left Blocked_precision']:.3f}, Recall={data['basic_metrics']['Left Blocked_recall']:.3f}, F1={data['basic_metrics']['Left Blocked_f1']:.3f}
• Right Blocked: Precision={data['basic_metrics']['Right Blocked_precision']:.3f}, Recall={data['basic_metrics']['Right Blocked_recall']:.3f}, F1={data['basic_metrics']['Right Blocked_f1']:.3f}
• Fully Blocked: Precision={data['basic_metrics']['Fully Blocked_precision']:.3f}, Recall={data['basic_metrics']['Fully Blocked_recall']:.3f}, F1={data['basic_metrics']['Fully Blocked_f1']:.3f}

Dataset Information:
• Total Samples: {len(data['y_true'])}
• Classes: 4 (Clear, Left Blocked, Right Blocked, Fully Blocked)
• Evaluation Type: 4-class path detection
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.title('Evaluation Summary Report', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to generate all graphs"""
    # Load data
    data_path = '/Users/lachiamine/Documents/python project/evaluation_results.json'
    data = load_evaluation_data(data_path)
    
    # Create output directory
    output_dir = Path('/Users/lachiamine/Documents/python project/evaluation_graphs')
    output_dir.mkdir(exist_ok=True)
    
    print("🎨 Generating Evaluation Graphs...")
    print(f"📊 Loaded data from {data_path}")
    print(f"💾 Saving graphs to {output_dir}")
    
    # Generate all graphs
    graphs = [
        ("Confusion Matrix", plot_confusion_matrix, "confusion_matrix.png"),
        ("Class Performance Metrics", plot_class_metrics, "class_metrics.png"),
        ("Prediction Distribution", plot_prediction_distribution, "prediction_distribution.png"),
        ("Overall Metrics", plot_overall_metrics, "overall_metrics.png"),
        ("Per-Class Accuracy", plot_class_accuracy, "class_accuracy.png"),
        ("Summary Report", create_summary_report, "summary_report.png")
    ]
    
    for title, plot_func, filename in graphs:
        print(f"📈 Creating {title}...")
        save_path = output_dir / filename
        plot_func(data, save_path)
        print(f"✅ Saved {filename}")
    
    print(f"\n🎉 All graphs generated successfully!")
    print(f"📁 Check the '{output_dir}' directory for all visualizations")
    
    # Print key metrics
    print(f"\n📋 Key Metrics Summary:")
    print(f"   Accuracy: {data['basic_metrics']['accuracy']:.3f} ({data['basic_metrics']['accuracy']*100:.1f}%)")
    print(f"   Best Performing Class: Left Blocked (F1: {data['basic_metrics']['Left Blocked_f1']:.3f})")
    print(f"   Worst Performing Class: Fully Blocked (F1: {data['basic_metrics']['Fully Blocked_f1']:.3f})")

if __name__ == "__main__":
    main()
