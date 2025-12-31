"""
Inference Script for Blind Path Detection System
"""

import cv2
import numpy as np
import tensorflow as tf
from config import *
from FinalProject.blind_path_detection.decision.decision_integrator import DecisionIntegrator


def load_model(model_path=None):
    """Load trained model"""
    if model_path is None:
        model_path = MODEL_PATH

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    return model


def preprocess_image(image):
    """Preprocess image for inference"""
    # Resize
    img_resized = cv2.resize(image, INPUT_SHAPE[:2])

    # Normalize
    img_norm = img_resized / 255.0

    # Add batch dimension
    img_tensor = np.expand_dims(img_norm, axis=0)

    return img_tensor


def predict_image(model, image, decision_integrator=None):
    """Predict single image"""
    # Preprocess
    img_tensor = preprocess_image(image)

    # Predict
    predictions = model.predict(img_tensor, verbose=0)[0]

    # Make decision
    if decision_integrator is None:
        decision_integrator = DecisionIntegrator()

    decision = decision_integrator.make_decision(
        probabilities=predictions,
        confidence=np.max(predictions)
    )

    return decision, predictions


def process_frame(frame, model, decision_integrator=None):
    """Process single video frame"""
    # Predict
    decision, probs = predict_image(model, frame, decision_integrator)

    return decision, probs


def run_image_inference(image_path, model_path=None):
    """Run inference on single image"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load image: {image_path}")
        return None

    # Load model
    model = load_model(model_path)

    # Create decision integrator
    decision_integrator = DecisionIntegrator()

    # Predict
    decision, probs = predict_image(model, image, decision_integrator)

    # Display results
    print("\n" + "=" * 50)
    print("Inference Results:")
    print("=" * 50)
    print(f"Image: {image_path}")
    print(f"Decision Level: {decision.get('level')}")
    print(f"Message: {decision.get('message')}")
    print(f"Risk Level: {decision.get('risk_level')}")
    print(f"Confidence: {decision.get('confidence'):.4f}")
    print(f"Probabilities:")
    print(f"  Clear: {probs[0]:.4f}")
    print(f"  Partial: {probs[1]:.4f}")
    print(f"  Full: {probs[2]:.4f}")

    return decision, probs


def main():
    """Main inference function"""
    import argparse

    parser = argparse.ArgumentParser(description='Blind Path Detection Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default=str(MODEL_PATH),
                        help='Model path')
    parser.add_argument('--mode', type=str, default='conservative',
                        choices=['conservative', 'balanced', 'aggressive'],
                        help='Safety mode')

    args = parser.parse_args()

    # Run inference
    run_image_inference(args.image, args.model)


if __name__ == "__main__":
    main()