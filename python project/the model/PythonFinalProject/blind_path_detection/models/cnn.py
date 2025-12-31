"""
CNN Model Definitions - Optimized for 20,000 images dataset
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from config import DROPOUT_RATE, BATCH_NORM_MOMENTUM


def build_cnn_v1(input_shape, num_classes, use_batchnorm=True):
    """
    Version 1: Deeper CNN optimized for 20,000 images
    """
    model = models.Sequential([
        # First convolution block
        layers.Conv2D(32, 3, padding='same', input_shape=input_shape),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM) if use_batchnorm else layers.Layer(),
        layers.ReLU(),
        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM) if use_batchnorm else layers.Layer(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(DROPOUT_RATE * 0.5),

        # Second convolution block
        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM) if use_batchnorm else layers.Layer(),
        layers.ReLU(),
        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM) if use_batchnorm else layers.Layer(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(DROPOUT_RATE * 0.6),

        # Third convolution block
        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM) if use_batchnorm else layers.Layer(),
        layers.ReLU(),
        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM) if use_batchnorm else layers.Layer(),
        layers.ReLU(),
        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM) if use_batchnorm else layers.Layer(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(DROPOUT_RATE * 0.7),

        # Fourth convolution block
        layers.Conv2D(256, 3, padding='same'),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM) if use_batchnorm else layers.Layer(),
        layers.ReLU(),
        layers.Conv2D(256, 3, padding='same'),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM) if use_batchnorm else layers.Layer(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),

        # Fully connected layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM) if use_batchnorm else layers.Layer(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(256, activation='relu'),
        layers.Dropout(DROPOUT_RATE * 0.8),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def build_cnn_v2(input_shape, num_classes):
    """
    Version 2: Lighter CNN for faster inference
    """
    model = models.Sequential([
        layers.Conv2D(32, 3, padding='same', input_shape=input_shape),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM),
        layers.ReLU(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM),
        layers.ReLU(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM),
        layers.ReLU(),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),

        layers.Conv2D(256, 3, padding='same'),
        layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model