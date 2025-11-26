"""
Emotion Inference Module for FER2013 ResNet-18 Model

This module provides inference functionality for the trained emotion recognition model.
It handles loading the model checkpoint and predicting emotions from facial images.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from PIL import Image

from src.models.fer_resnet import FerResNet18
from src.data.fer2013_dataset import CLASS_NAMES, NUM_CLASSES


class EmotionPredictor:
    """
    Emotion predictor class that loads a trained model and performs inference.
    
    Attributes:
        model: Loaded FerResNet18 model
        device: Device (CPU or CUDA) where model is running
        class_names: List of emotion class names
        checkpoint_path: Path to the loaded checkpoint
    """
    
    def __init__(
        self,
        checkpoint_path: str = 'models/fer_resnet18_best.pth',
        use_gpu_if_available: bool = True
    ):
        """
        Initialize the emotion predictor.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            use_gpu_if_available: Whether to use GPU if available
        """
        self.checkpoint_path = checkpoint_path
        
        # Determine device
        if use_gpu_if_available and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
        # Load model
        self.model, self.class_names = self._load_model()
        self.model.eval()  # Set to evaluation mode
        
        print(f"Model loaded successfully from: {checkpoint_path}")
        print(f"Classes: {self.class_names}")
    
    def _load_model(self) -> Tuple[FerResNet18, list]:
        """
        Load the trained model from checkpoint.
        
        Returns:
            model: Loaded FerResNet18 model
            class_names: List of class names from checkpoint
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Please train the model first using: python -m src.training.train_fer_resnet"
            )
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Get class names and number of classes from checkpoint
        class_names = checkpoint.get('class_names', CLASS_NAMES)
        num_classes = checkpoint.get('num_classes', NUM_CLASSES)
        
        # Create model with same architecture
        model = FerResNet18(num_classes=num_classes, input_channels=1)
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to device
        model = model.to(self.device)
        
        # Print checkpoint info
        epoch = checkpoint.get('epoch', 'unknown')
        val_acc = checkpoint.get('val_accuracy', 'unknown')
        print(f"\nCheckpoint Info:")
        print(f"  Epoch: {epoch}")
        print(f"  Validation Accuracy: {val_acc}")
        
        return model, class_names
    
    def preprocess_face(
        self,
        face_image: np.ndarray,
        target_size: Tuple[int, int] = (48, 48)
    ) -> torch.Tensor:
        """
        Preprocess a face image for model input.
        
        Args:
            face_image: Face image (can be grayscale or BGR)
            target_size: Target size for resizing (height, width)
            
        Returns:
            Preprocessed tensor of shape (1, 1, 48, 48)
        """
        # Convert BGR to grayscale if needed
        if len(face_image.shape) == 3:
            if face_image.shape[2] == 3:
                face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_image[:, :, 0]  # Take first channel if not 3 channels
        else:
            face_gray = face_image
        
        # Resize to target size
        face_resized = cv2.resize(face_gray, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Convert to PIL Image for consistency with training transforms
        face_pil = Image.fromarray((face_normalized * 255).astype(np.uint8), mode='L')
        
        # Convert to tensor and add batch and channel dimensions
        # Shape: (48, 48) -> (1, 48, 48) -> (1, 1, 48, 48)
        face_tensor = torch.from_numpy(np.array(face_pil, dtype=np.float32) / 255.0)
        face_tensor = face_tensor.unsqueeze(0).unsqueeze(0)
        
        return face_tensor
    
    def predict(
        self,
        face_image: np.ndarray
    ) -> Tuple[str, int, np.ndarray]:
        """
        Predict emotion from a face image.
        
        Args:
            face_image: Face image (grayscale or BGR)
            
        Returns:
            label_str: Predicted emotion label string
            label_idx: Predicted emotion label index
            probabilities: Class probabilities as numpy array
        """
        # Preprocess image
        face_tensor = self.preprocess_face(face_image)
        face_tensor = face_tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get prediction
        prob_values = probabilities.cpu().numpy()[0]
        label_idx = int(np.argmax(prob_values))
        label_str = self.class_names[label_idx]
        
        return label_str, label_idx, prob_values
    
    def predict_emotion_from_frame(
        self,
        frame_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> Tuple[str, int, np.ndarray]:
        """
        Predict emotion from a frame with a face bounding box.
        
        Args:
            frame_bgr: OpenCV BGR image (full frame)
            face_bbox: Face bounding box as (x, y, w, h)
            
        Returns:
            label_str: Predicted emotion label string
            label_idx: Predicted emotion label index
            probabilities: Class probabilities as numpy array
        """
        x, y, w, h = face_bbox
        
        # Add boundary checks
        height, width = frame_bgr.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)
        
        # Crop face from frame
        face_crop = frame_bgr[y:y+h, x:x+w]
        
        if face_crop.size == 0:
            raise ValueError("Invalid face bounding box - crop is empty")
        
        # Predict emotion
        return self.predict(face_crop)


def predict_emotion_from_frame(
    frame_bgr: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    checkpoint_path: str = 'models/fer_resnet18_best.pth'
) -> Tuple[str, int, np.ndarray]:
    """
    Convenience function to predict emotion from a frame.
    
    This creates a new predictor each time. For repeated predictions,
    create an EmotionPredictor instance and reuse it.
    
    Args:
        frame_bgr: OpenCV BGR image
        face_bbox: Face bounding box as (x, y, w, h)
        checkpoint_path: Path to model checkpoint
        
    Returns:
        label_str: Predicted emotion label string
        label_idx: Predicted emotion label index
        probabilities: Class probabilities as numpy array
        
    Example:
        >>> frame = cv2.imread('image.jpg')
        >>> bbox = (100, 100, 150, 150)
        >>> emotion, idx, probs = predict_emotion_from_frame(frame, bbox)
        >>> print(f"Emotion: {emotion}, Confidence: {probs[idx]:.2%}")
    """
    predictor = EmotionPredictor(checkpoint_path)
    return predictor.predict_emotion_from_frame(frame_bgr, face_bbox)


def load_sample_from_dataset(
    csv_path: str = 'data/fer2013.csv',
    usage: str = 'PrivateTest',
    index: int = 0
) -> Tuple[np.ndarray, int]:
    """
    Load a sample image from the FER2013 dataset for testing.
    
    Args:
        csv_path: Path to fer2013.csv
        usage: Dataset split ('Training', 'PublicTest', or 'PrivateTest')
        index: Index of sample to load
        
    Returns:
        image: Grayscale image as numpy array
        true_label: True emotion label
    """
    import pandas as pd
    from src.data.fer2013_dataset import CLASS_MAPPING
    
    # Load dataset
    df = pd.read_csv(csv_path)
    df_subset = df[df['Usage'] == usage]
    
    if index >= len(df_subset):
        raise ValueError(f"Index {index} out of range for {usage} set (size: {len(df_subset)})")
    
    # Get sample
    row = df_subset.iloc[index]
    pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
    image = pixels.reshape(48, 48)
    
    # Map original label to target label
    original_label = int(row['emotion'])
    target_label = CLASS_MAPPING[original_label]
    
    return image, target_label


if __name__ == "__main__":
    """
    Test block to verify inference functionality.
    """
    print("=" * 80)
    print("Emotion Inference Test")
    print("=" * 80)
    
    # Check if checkpoint exists
    checkpoint_path = 'models/fer_resnet18_best.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Please train the model first:")
        print("  python -m src.training.train_fer_resnet --epochs 20")
        print("\nFor testing without training, we'll create a dummy model...")
        
        # Create a dummy model for testing the inference pipeline
        from src.models.fer_resnet import FerResNet18
        dummy_model = FerResNet18(num_classes=4)
        os.makedirs('models', exist_ok=True)
        
        dummy_checkpoint = {
            'epoch': 0,
            'model_state_dict': dummy_model.state_dict(),
            'val_accuracy': 0.0,
            'class_names': CLASS_NAMES,
            'num_classes': NUM_CLASSES
        }
        torch.save(dummy_checkpoint, checkpoint_path)
        print(f"Created dummy checkpoint for testing: {checkpoint_path}")
    
    # Initialize predictor
    print("\n" + "=" * 80)
    print("Initializing Emotion Predictor...")
    print("=" * 80)
    
    predictor = EmotionPredictor(checkpoint_path)
    
    # Test 1: Predict from synthetic random image
    print("\n" + "=" * 80)
    print("Test 1: Random Synthetic Image")
    print("=" * 80)
    
    # Create random 48x48 grayscale image
    random_face = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
    
    emotion, label_idx, probabilities = predictor.predict(random_face)
    
    print(f"\nPrediction Results:")
    print(f"  Predicted Emotion: {emotion}")
    print(f"  Label Index: {label_idx}")
    print(f"\nClass Probabilities:")
    for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
        marker = " <--" if i == label_idx else ""
        print(f"  {class_name:8s}: {prob:.4f} ({prob*100:.2f}%){marker}")
    
    # Test 2: Predict from frame with bounding box
    print("\n" + "=" * 80)
    print("Test 2: Frame with Bounding Box")
    print("=" * 80)
    
    # Create a synthetic frame (640x480 BGR image)
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    face_bbox = (200, 150, 200, 200)  # (x, y, w, h)
    
    print(f"Frame shape: {frame.shape}")
    print(f"Face bbox: {face_bbox}")
    
    emotion, label_idx, probabilities = predictor.predict_emotion_from_frame(frame, face_bbox)
    
    print(f"\nPrediction Results:")
    print(f"  Predicted Emotion: {emotion}")
    print(f"  Label Index: {label_idx}")
    print(f"  Confidence: {probabilities[label_idx]:.4f} ({probabilities[label_idx]*100:.2f}%)")
    
    # Test 3: Load from FER2013 dataset if available
    csv_path = 'data/fer2013.csv'
    if os.path.exists(csv_path):
        print("\n" + "=" * 80)
        print("Test 3: Real FER2013 Sample")
        print("=" * 80)
        
        try:
            # Load a sample from test set
            sample_image, true_label = load_sample_from_dataset(
                csv_path, usage='PrivateTest', index=0
            )
            
            print(f"Loaded sample from FER2013 test set")
            print(f"Image shape: {sample_image.shape}")
            print(f"True label: {CLASS_NAMES[true_label]}")
            
            # Predict
            emotion, label_idx, probabilities = predictor.predict(sample_image)
            
            print(f"\nPrediction Results:")
            print(f"  Predicted: {emotion}")
            print(f"  True label: {CLASS_NAMES[true_label]}")
            print(f"  Match: {'✓ Correct' if label_idx == true_label else '✗ Incorrect'}")
            print(f"  Confidence: {probabilities[label_idx]:.4f} ({probabilities[label_idx]*100:.2f}%)")
            
            print(f"\nAll Probabilities:")
            for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
                marker = " <-- PREDICTED" if i == label_idx else ""
                true_marker = " <-- TRUE" if i == true_label else ""
                print(f"  {class_name:8s}: {prob:.4f} ({prob*100:.2f}%){marker}{true_marker}")
                
        except Exception as e:
            print(f"Error loading FER2013 sample: {e}")
    else:
        print(f"\n{csv_path} not found - skipping real data test")
    
    # Performance test
    print("\n" + "=" * 80)
    print("Performance Test")
    print("=" * 80)
    
    import time
    
    # Warm-up
    for _ in range(5):
        test_face = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
        _ = predictor.predict(test_face)
    
    # Benchmark
    num_iterations = 100
    start_time = time.time()
    
    for _ in range(num_iterations):
        test_face = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
        _ = predictor.predict(test_face)
    
    elapsed_time = time.time() - start_time
    fps = num_iterations / elapsed_time
    avg_time_ms = (elapsed_time / num_iterations) * 1000
    
    print(f"\nInference Performance:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Total time: {elapsed_time:.2f}s")
    print(f"  Average time per prediction: {avg_time_ms:.2f}ms")
    print(f"  Throughput: {fps:.2f} predictions/second")
    
    print("\n" + "=" * 80)
    print("Inference test completed successfully!")
    print("=" * 80)
    
    print("\nUsage Example:")
    print("-" * 80)
    print("# For single prediction:")
    print("from src.inference.emotion_inference import EmotionPredictor")
    print("predictor = EmotionPredictor('models/fer_resnet18_best.pth')")
    print("emotion, idx, probs = predictor.predict_emotion_from_frame(frame, (x, y, w, h))")
    print(f"print(f'Emotion: {{emotion}}, Confidence: {{probs[idx]:.2%}}')")
