"""
Plotting Utilities for Blind Path Detection System
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from config import *


def plot_training_curves(history, experiment_name):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curve
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy curve
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision curve
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Precision Curve')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Recall curve
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Recall Curve')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.suptitle(f'Training Curves - {experiment_name}', fontsize=16)
    plt.tight_layout()

    # Save plot
    plot_path = OUTPUT_DIR / f"training_curves_{experiment_name}.png"
    plt.savefig(str(plot_path), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved: {plot_path}")


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=PLOT_FIGSIZE)
    sns.heatmap(cm, annot=True, fmt='d', cmap=CMAP,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - Blind Path Detection', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    return cm


def plot_class_distribution(labels, save_path=None):
    """Plot class distribution"""
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=PLOT_FIGSIZE)
    bars = plt.bar([CLASS_NAMES[i] for i in unique], counts,
                   color=['green', 'orange', 'red'])

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{count}', ha='center', va='bottom', fontsize=12)

    plt.title('Dataset Class Distribution', fontsize=16, pad=20)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_error_analysis(error_stats, save_path=None):
    """Plot error analysis chart"""
    categories = list(error_stats.keys())
    counts = list(error_stats.values())

    plt.figure(figsize=PLOT_FIGSIZE)
    bars = plt.barh(categories, counts, color=['red', 'orange', 'yellow'])

    # Add count labels
    for bar, count in zip(bars, counts):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{count}', ha='left', va='center', fontsize=12)

    plt.title('Error Sample Analysis', fontsize=16, pad=20)
    plt.xlabel('Error Count', fontsize=12)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()