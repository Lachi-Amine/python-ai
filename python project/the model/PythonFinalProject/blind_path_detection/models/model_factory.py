"""
Model Factory for Blind Path Detection System
"""

from FinalProject.blind_path_detection.models.cnn import build_cnn_v1, build_cnn_v2
from FinalProject.blind_path_detection.models.mobilenet import build_mobilenet


def build_model(model_type, input_shape, num_classes, use_transfer=True, **kwargs):
    """
    Model factory function

    Args:
        model_type: Type of model to build ("cnn_v1", "cnn_v2", "mobilenet")
        input_shape: Input image shape
        num_classes: Number of output classes
        use_transfer: Whether to use transfer learning (for MobileNet)
        **kwargs: Additional model-specific arguments

    Returns:
        Compiled Keras model
    """
    if model_type == "cnn_v1":
        return build_cnn_v1(input_shape, num_classes, **kwargs)
    elif model_type == "cnn_v2":
        return build_cnn_v2(input_shape, num_classes)
    elif model_type == "mobilenet":
        return build_mobilenet(input_shape, num_classes, use_transfer, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")