"""
Dataset Utilities for Blind Path Detection System
"""

import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from config import *


def load_dataset(data_dir=None, validation_split=0.2, augment=True):
    """
    Load and preprocess dataset

    Args:
        data_dir: Directory containing dataset
        validation_split: Fraction of data to use for validation
        augment: Whether to apply data augmentation

    Returns:
        train_ds: Training dataset
        val_ds: Validation dataset
    """
    data_dir = data_dir or DATA_DIR

    # Data augmentation configuration
    if augment:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ])

    # Load training dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # Load validation dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # Normalization layer
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    def preprocess(image, label):
        image = normalization_layer(image)
        if augment:
            image = augmentation(image, training=True)
        return image, label

    train_ds = train_ds.map(preprocess)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def compute_class_weights(dataset):
    """
    Compute class weights for imbalanced data

    Args:
        dataset: TensorFlow dataset with categorical labels

    Returns:
        dict: Class weights for each class
    """
    labels = []
    for _, batch_labels in dataset:
        labels.extend(np.argmax(batch_labels.numpy(), axis=1))

    unique_labels = np.unique(labels)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=labels
    )

    return dict(zip(unique_labels, weights))