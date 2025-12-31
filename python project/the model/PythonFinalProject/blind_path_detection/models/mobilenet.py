"""
MobileNet Model Definition
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from config import DROPOUT_RATE


def build_mobilenet(input_shape, num_classes, use_transfer=True, freeze_layers=120):
    """
    Build MobileNet model with transfer learning support

    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        use_transfer: Whether to use transfer learning
        freeze_layers: Number of layers to freeze (if using transfer learning)
    """
    if use_transfer:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    else:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )

    # Fine-tuning strategy
    if use_transfer:
        # Freeze early layers, fine-tune later layers
        for layer in base_model.layers[:freeze_layers]:
            layer.trainable = False
        for layer in base_model.layers[freeze_layers:]:
            layer.trainable = True
    else:
        # Train all layers
        base_model.trainable = True

    # Add custom classification head
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE * 0.8)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model