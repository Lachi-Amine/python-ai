#%%
# =========================
# Configuration
# =========================

from pathlib import Path

# Dataset
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 4
CLASS_NAMES = [
    "Clear",
    "Left Blocked",
    "Right Blocked",
    "Fully Blocked"
]

DATA_DIR = Path(r"C:\Users\lenovo\Desktop\PythonCode\PythonFinalProject2\dataset")

# Normalization
NORMALIZATION_MODE = "mobilenet"  # "rescale" | "mobilenet"

# Model
MODEL_TYPE = "mobilenet"
USE_TRANSFER = True
FREEZE_LAYERS = 120
DROPOUT_RATE = 0.5
BATCH_NORM_MOMENTUM = 0.9

# Regularization
L2_WEIGHT_DECAY = 1e-4

# Training
BATCH_SIZE = 32
EPOCHS = 150
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4
OPTIMIZER_TYPE = "adam"

EARLY_STOPPING_PATIENCE = 12
LR_PATIENCE = 4
LR_FACTOR = 0.3

USE_CLASS_WEIGHTS = True
USE_DATA_AUGMENTATION = True

# Output directories
BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"
MODEL_PATH = BASE_DIR / "best_model.h5"

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
import time

import numpy as np
import tensorflow as tf

from sklearn.utils import class_weight
from tensorflow.keras import layers, models

def _get_normalization_fn():
    if NORMALIZATION_MODE == "rescale":
        return tf.keras.layers.Rescaling(1.0 / 255)

    if NORMALIZATION_MODE == "mobilenet":
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

        def fn(x):
            x = tf.cast(x, tf.float32)
            return preprocess(x)

        return fn

    raise ValueError("Unknown NORMALIZATION_MODE")

def load_dataset(data_dir, validation_split=0.2, augment=True):

    if augment:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ])

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    normalize = _get_normalization_fn()

    def preprocess_train(x, y):
        if augment:
            x = augmentation(x, training=True)
        x = normalize(x)
        return x, y

    def preprocess_val(x, y):
        x = normalize(x)
        return x, y

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds
        .map(preprocess_train, num_parallel_calls=AUTOTUNE)
        .shuffle(1000)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds
        .map(preprocess_val, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds

def compute_class_weights(dataset):
    labels = []

    for _, y in dataset:
        labels.extend(np.argmax(y.numpy(), axis=1))

    classes = np.unique(labels)

    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    return dict(zip(classes, weights))

def _get_l2():
    return tf.keras.regularizers.l2(L2_WEIGHT_DECAY) if L2_WEIGHT_DECAY > 0 else None

def build_mobilenet(input_shape, num_classes, use_transfer=True, freeze_layers=120):
    l2_reg = _get_l2()

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet" if use_transfer else None,
    )

    if use_transfer:
        for layer in base_model.layers[:freeze_layers]:
            layer.trainable = False
        for layer in base_model.layers[freeze_layers:]:
            layer.trainable = True

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(DROPOUT_RATE * 0.8)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(base_model.input, outputs)

def get_optimizer(name, lr):
    if name == "adam":
        return tf.keras.optimizers.Adam(lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(lr)
    raise ValueError("Unknown optimizer")

train_ds, val_ds = load_dataset(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    augment=USE_DATA_AUGMENTATION
)

class_weights = compute_class_weights(train_ds) if USE_CLASS_WEIGHTS else None
print("Class weights:", class_weights)

model = build_mobilenet(
    INPUT_SHAPE,
    NUM_CLASSES,
    use_transfer=USE_TRANSFER,
    freeze_layers=FREEZE_LAYERS
)

model.compile(
    optimizer=get_optimizer(OPTIMIZER_TYPE, LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        monitor="val_loss"
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=LR_PATIENCE,
        factor=LR_FACTOR,
        min_lr=1e-7
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True,
        monitor="val_loss"
    ),
    tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
]

start = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks  # ✅ ADD THIS LINE
)

print(f"Training time: {time.time() - start:.2f}s")

final_path = OUTPUT_DIR / "final_model.keras"
model.save(final_path)
print("Saved:", final_path)