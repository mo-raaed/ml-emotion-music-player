"""
Static Image Emotion Detection Demo

This script demonstrates emotion detection on static images.
It detects faces and predicts emotions, then displays the results.

Usage:
    python demo_image.py <image_path>
    python demo_image.py test_image.jpg
"""

import sys
import cv2
import numpy as np
import os
from pathlib import Path

from src.inference.emotion_inference import EmotionPredictor
from src.data.fer2013_dataset import CLASS_NAMES


# Colors for each emotion (BGR format)
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'sad': (255, 0, 0),        # Blue
    'happy': (0, 255, 0),      # Green
    'neutral': (255, 255, 0),  # Cyan
}


def detect_and_predict_emotions(
    image_path: str,
    predictor: EmotionPredictor,
    save_output: bool = True
) -> None:
    """
    Detect faces and predict emotions in an image.
    
    Args:
        image_path: Path to input image
        predictor: EmotionPredictor instance
        save_output: Whether to save the output image
    """
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"\nProcessing image: {image_path}")
    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Load face detector
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )
    
    print(f"Detected {len(faces)} face(s)")
    
    # Process each face
    for i, (x, y, w, h) in enumerate(faces):
        print(f"\nFace {i+1}:")
        print(f"  Location: ({x}, {y}), Size: {w}x{h}")
        
        try:
            # Predict emotion
            emotion, label_idx, probabilities = predictor.predict_emotion_from_frame(
                frame, (x, y, w, h)
            )
            
            print(f"  Predicted emotion: {emotion}")
            print(f"  Confidence: {probabilities[label_idx]:.2%}")
            print(f"  All probabilities:")
            for class_name, prob in zip(CLASS_NAMES, probabilities):
                print(f"    {class_name:8s}: {prob:.4f} ({prob*100:.2f}%)")
            
            # Draw on frame
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            
            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Draw emotion label
            label = f"{emotion}: {probabilities[label_idx]:.1%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            
            # Draw background for label
            cv2.rectangle(
                frame,
                (x, y - label_size[1] - 15),
                (x + label_size[0] + 10, y),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )
            
        except Exception as e:
            print(f"  Error processing face: {e}")
    
    # Display result
    cv2.imshow('Emotion Detection Result', frame)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save output
    if save_output and len(faces) > 0:
        output_path = Path(image_path).stem + "_emotion_output.jpg"
        cv2.imwrite(output_path, frame)
        print(f"\nOutput saved to: {output_path}")


def main():
    """
    Main function.
    """
    print("=" * 80)
    print("Static Image Emotion Detection Demo")
    print("=" * 80)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: python demo_image.py <image_path>")
        print("\nExample:")
        print("  python demo_image.py test_image.jpg")
        print("  python demo_image.py path/to/your/photo.png")
        
        # Try to use a sample from dataset if available
        csv_path = 'data/fer2013.csv'
        if os.path.exists(csv_path):
            print("\n" + "=" * 80)
            print("No image provided. Loading sample from FER2013 dataset...")
            print("=" * 80)
            
            from src.inference.emotion_inference import load_sample_from_dataset
            
            # Load a sample
            sample_image, true_label = load_sample_from_dataset(
                csv_path, usage='PrivateTest', index=42
            )
            
            # Save as temporary file
            temp_image_path = 'temp_sample.jpg'
            cv2.imwrite(temp_image_path, sample_image)
            print(f"Created temporary image: {temp_image_path}")
            print(f"True emotion: {CLASS_NAMES[true_label]}")
            
            image_path = temp_image_path
        else:
            return
    else:
        image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\nError: Image not found: {image_path}")
        return
    
    # Check if model checkpoint exists
    checkpoint_path = 'models/fer_resnet18_best.pth'
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Model checkpoint not found: {checkpoint_path}")
        print("\nPlease train the model first:")
        print("  python -m src.training.train_fer_resnet --epochs 20")
        return
    
    # Load predictor
    print("\nLoading emotion predictor...")
    predictor = EmotionPredictor(checkpoint_path)
    
    # Process image
    print("\n" + "=" * 80)
    detect_and_predict_emotions(image_path, predictor)
    
    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
