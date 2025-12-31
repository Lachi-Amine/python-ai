"""
Training Script for Blind Path Detection System
"""
from config import *
import tensorflow as tf
import time

from FinalProject.blind_path_detection.models.model_factory import build_model
from FinalProject.blind_path_detection.utils.dataset import load_dataset, compute_class_weights
from FinalProject.blind_path_detection.utils.logger import TrainingLogger
from FinalProject.blind_path_detection.utils.plots import plot_training_curves


def get_optimizer(optimizer_type, lr):
    """Get optimizer based on configuration"""
    if optimizer_type == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_type == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=0.9,
            nesterov=True
        )
    elif optimizer_type == "rmsprop":
        return tf.keras.optimizers.RMSprop(
            learning_rate=lr,
            rho=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def train_experiment(model_type, use_transfer, optimizer_type, experiment_name):
    """Execute single training experiment"""
    print(f"\n{'=' * 60}")
    print(f"Starting experiment: {experiment_name}")
    print(f"Model: {model_type}, Transfer Learning: {use_transfer}, Optimizer: {optimizer_type}")
    print(f"{'=' * 60}")

    # Load data
    train_ds, val_ds = load_dataset()

    # Compute class weights
    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_ds)
        print(f"Class weights: {class_weights}")

    # Build model
    model = build_model(
        model_type=model_type,
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        use_transfer=use_transfer
    )

    # Compile model
    model.compile(
        optimizer=get_optimizer(optimizer_type, LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(LOG_DIR / experiment_name),
            histogram_freq=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(LOG_DIR / f"{experiment_name}_history.csv")
        )
    ]

    # Train model
    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    training_time = time.time() - start_time

    # Collect results
    results = {
        "training_time": training_time,
        "final_epoch": len(history.history['loss']),
        "final_loss": history.history['loss'][-1],
        "final_accuracy": history.history['accuracy'][-1],
        "final_val_loss": history.history['val_loss'][-1],
        "final_val_accuracy": history.history['val_accuracy'][-1],
        "best_val_accuracy": max(history.history['val_accuracy']),
        "best_val_loss": min(history.history['val_loss'])
    }

    print(f"\nExperiment completed: {experiment_name}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")

    return model, history, results


def run_training_comparison():
    """Run complete training comparison experiments"""

    # Define experiment configurations
    experiments = [
        # Optimal configuration
        {
            "name": "optimal_mobilenet_transfer",
            "model_type": "mobilenet",
            "use_transfer": True,
            "optimizer": "adam"
        },
        # Medium configuration
        {
            "name": "cnn_v1_baseline",
            "model_type": "cnn_v1",
            "use_transfer": False,
            "optimizer": "adam"
        },
        # Poor configuration (for comparison)
        {
            "name": "cnn_v2_sgd_no_tuning",
            "model_type": "cnn_v2",
            "use_transfer": False,
            "optimizer": "sgd"
        }
    ]

    # Initialize logger
    logger = TrainingLogger(LOG_DIR)

    # Log configuration
    config_dict = {
        "input_shape": INPUT_SHAPE,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "dropout_rate": DROPOUT_RATE
    }
    logger.log_config(config_dict)

    all_results = {}

    # Run all experiments
    for exp in experiments:
        model, history, results = train_experiment(
            model_type=exp["model_type"],
            use_transfer=exp["use_transfer"],
            optimizer_type=exp["optimizer"],
            experiment_name=exp["name"]
        )

        # Log experiment results
        logger.log_experiment(
            experiment_name=exp["name"],
            params=exp,
            results=results
        )

        # Save best model for each experiment
        model_path = LOG_DIR / f"best_model_{exp['name']}.h5"
        model.save(str(model_path))

        all_results[exp["name"]] = results

        # Generate training curves
        plot_training_curves(history, exp["name"])

    # Generate comparison report
    generate_comparison_report(all_results)

    return all_results


def generate_comparison_report(all_results):
    """Generate comparison experiment report"""
    report_path = OUTPUT_DIR / "model_comparison_report.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Blind Path Detection System - Model Comparison Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("Experiment Configuration:\n")
        f.write(f"   Input Shape: {INPUT_SHAPE}\n")
        f.write(f"   Number of Classes: {NUM_CLASSES}\n")
        f.write(f"   Batch Size: {BATCH_SIZE}\n")
        f.write(f"   Learning Rate: {LEARNING_RATE}\n")
        f.write(f"   Dropout Rate: {DROPOUT_RATE}\n")
        f.write(f"   Use Class Weights: {USE_CLASS_WEIGHTS}\n\n")

        f.write("=" * 70 + "\n")
        f.write("Experiment Results Comparison:\n")
        f.write("=" * 70 + "\n\n")

        # Print header
        header = f"{'Experiment Name':<25} {'Training Time(s)':<15} {'Best Val Acc':<15} {'Best Val Loss':<15} {'Final Epoch':<10}"
        f.write(header + "\n")
        f.write("-" * 85 + "\n")

        # Print each experiment's results
        for exp_name, results in all_results.items():
            row = f"{exp_name:<25} {results['training_time']:<15.1f} "
            row += f"{results['best_val_accuracy']:<15.4f} {results['best_val_loss']:<15.4f} "
            row += f"{results['final_epoch']:<10}"
            f.write(row + "\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Conclusions & Recommendations:\n")
        f.write("=" * 70 + "\n\n")

        # Find best model
        best_exp = max(all_results.items(), key=lambda x: x[1]['best_val_accuracy'])
        f.write(f"1. Best Model: {best_exp[0]}\n")
        f.write(f"   Validation Accuracy: {best_exp[1]['best_val_accuracy']:.4f}\n")
        f.write(f"   Training Time: {best_exp[1]['training_time']:.1f} seconds\n\n")

        f.write("2. Model Selection Recommendations:\n")
        f.write("   - For maximum accuracy: Choose MobileNet with transfer learning\n")
        f.write("   - For speed: Choose CNN_V2\n")
        f.write("   - For balance: Choose CNN_V1\n\n")

        f.write("3. Training Strategy Recommendations:\n")
        f.write("   - Use class weights for imbalanced data\n")
        f.write("   - Use Adam optimizer\n")
        f.write("   - Apply data augmentation\n")
        f.write("   - Use early stopping to prevent overfitting\n\n")

        f.write("4. Deployment Recommendations:\n")
        f.write("   - Use best model for production\n")
        f.write("   - Consider lightweight models for edge devices\n")
        f.write("   - Regularly retrain model with new data\n")

    print(f"\nComparison report saved: {report_path}")


def main():
    """Main training function"""
    print("Starting Blind Path Detection System Training...")

    # Run complete training comparison experiments
    results = run_training_comparison()

    print("\nTraining completed!")
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"Training logs saved to: {LOG_DIR}")
    print(f"Output results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()