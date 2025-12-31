"""
Grad-CAM Visualization Utilities for Blind Path Detection System
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from config import *


class GradCAM:
    """Grad-CAM visualization utility class"""

    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM

        Args:
            model: Pre-trained Keras model
            layer_name: Target layer name. If None, automatically finds last convolutional layer
        """
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()

        # Create gradient model
        self.grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )

    def _find_target_layer(self):
        """Automatically find last convolutional layer"""
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:  # Convolutional layer
                return layer.name
        raise ValueError("No convolutional layer found")

    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap

        Args:
            image: Input image (1, H, W, 3)
            class_idx: Target class index. If None, uses predicted class
            eps: Small value to prevent division by zero

        Returns:
            heatmap: Heatmap
            preds: Prediction probabilities
        """
        # Forward pass
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)

            if class_idx is None:
                class_idx = tf.argmax(predictions[0])

            loss = predictions[:, class_idx]

        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weighted feature maps
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

        # Normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)

        return heatmap.numpy(), predictions[0].numpy()

    def overlay_heatmap(self, original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image

        Args:
            original_img: Original image (H, W, 3)
            heatmap: Heatmap
            alpha: Overlay transparency
            colormap: OpenCV colormap

        Returns:
            overlay: Overlayed image
        """
        # Resize heatmap
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

        # Convert to 8-bit and apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)

        # Overlay
        overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)

        return overlay

    def visualize(self, image, class_idx=None, save_path=None):
        """
        Complete visualization pipeline

        Args:
            image: Original image
            class_idx: Target class index
            save_path: Save path

        Returns:
            fig: Matplotlib figure
        """
        # Preprocess image
        if len(image.shape) == 3:
            img_tensor = np.expand_dims(image, axis=0)
        else:
            img_tensor = image

        # Compute heatmap
        heatmap, preds = self.compute_heatmap(img_tensor, class_idx)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        if len(image.shape) == 4:
            display_img = image[0]
        else:
            display_img = image

        axes[0].imshow(display_img)
        axes[0].set_title(f"Original Image\nPrediction: {CLASS_NAMES[np.argmax(preds)]} ({np.max(preds):.2f})")
        axes[0].axis('off')

        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')

        # Overlay image
        if display_img.dtype == np.float32:
            display_img = (display_img * 255).astype(np.uint8)

        overlay = self.overlay_heatmap(display_img, heatmap)
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Overlay Visualization")
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=PLOT_DPI, bbox_inches='tight')

        return fig, heatmap, preds

    def batch_visualize(self, images, labels, save_dir=None, num_samples=10):
        """
        Batch visualize correct and incorrect samples

        Args:
            images: Image batch
            labels: True labels
            save_dir: Save directory
            num_samples: Number of samples to display per category
        """
        from sklearn.metrics import accuracy_score

        # Predictions
        preds = self.model.predict(images)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.argmax(labels, axis=1)

        # Compute accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Batch Accuracy: {accuracy:.4f}")

        # Create save directories
        if save_dir:
            correct_dir = Path(save_dir) / "correct"
            error_dir = Path(save_dir) / "error"
            correct_dir.mkdir(parents=True, exist_ok=True)
            error_dir.mkdir(parents=True, exist_ok=True)

        # Statistics for correct and incorrect samples
        correct_indices = np.where(y_pred == y_true)[0]
        error_indices = np.where(y_pred != y_true)[0]

        print(f"Correct Samples: {len(correct_indices)}")
        print(f"Error Samples: {len(error_indices)}")

        # Visualize correct samples
        print("\nVisualizing correct samples...")
        for i, idx in enumerate(correct_indices[:min(num_samples, len(correct_indices))]):
            if save_dir:
                save_path = correct_dir / f"correct_{i}_true_{CLASS_NAMES[y_true[idx]]}_pred_{CLASS_NAMES[y_pred[idx]]}.png"
            else:
                save_path = None

            fig, _, _ = self.visualize(
                images[idx:idx + 1],
                save_path=save_path
            )
            plt.close(fig)

        # Visualize error samples
        print("Visualizing error samples...")
        for i, idx in enumerate(error_indices[:min(num_samples, len(error_indices))]):
            if save_dir:
                save_path = error_dir / f"error_{i}_true_{CLASS_NAMES[y_true[idx]]}_pred_{CLASS_NAMES[y_pred[idx]]}.png"
            else:
                save_path = None

            fig, _, _ = self.visualize(
                images[idx:idx + 1],
                save_path=save_path
            )
            plt.close(fig)

        return correct_indices, error_indices