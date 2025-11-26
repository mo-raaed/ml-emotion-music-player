"""
Inference module for real-time emotion prediction.
"""

from .emotion_inference import (
    EmotionPredictor,
    predict_emotion_from_frame,
    load_sample_from_dataset
)

__all__ = [
    'EmotionPredictor',
    'predict_emotion_from_frame',
    'load_sample_from_dataset',
]
