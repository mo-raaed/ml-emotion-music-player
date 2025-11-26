"""
Real-time Emotion Detection Demo using Webcam

This script demonstrates real-time emotion detection using your webcam.
It detects faces using OpenCV's Haar Cascade and predicts emotions using
the trained ResNet-18 model.

Usage:
    python demo_webcam.py
    
Controls:
    - Press 'q' to quit
    - Press 's' to save a screenshot
"""

import cv2
import numpy as np
import os
from datetime import datetime

from src.inference.emotion_inference import EmotionPredictor
from src.data.fer2013_dataset import CLASS_NAMES


# Colors for each emotion (BGR format)
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'sad': (255, 0, 0),        # Blue
    'happy': (0, 255, 0),      # Green
    'neutral': (255, 255, 0),  # Cyan
}


def draw_emotion_info(
    frame: np.ndarray,
    face_bbox: tuple,
    emotion: str,
    probabilities: np.ndarray,
    class_names: list
) -> np.ndarray:
    """
    Draw emotion prediction information on the frame.
    
    Args:
        frame: Input frame
        face_bbox: Face bounding box (x, y, w, h)
        emotion: Predicted emotion label
        probabilities: Class probabilities
        class_names: List of class names
        
    Returns:
        Frame with drawn information
    """
    x, y, w, h = face_bbox
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    confidence = probabilities[class_names.index(emotion)]
    
    # Draw face bounding box
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Draw emotion label and confidence
    label = f"{emotion}: {confidence:.2%}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    
    # Draw background for label
    cv2.rectangle(
        frame,
        (x, y - label_size[1] - 10),
        (x + label_size[0], y),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        frame,
        label,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    
    # Draw probability bars on the side
    bar_x = frame.shape[1] - 200
    bar_y_start = 30
    bar_height = 25
    bar_max_width = 180
    
    # Sort probabilities by value
    sorted_emotions = sorted(
        zip(class_names, probabilities),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (class_name, prob) in enumerate(sorted_emotions):
        bar_y = bar_y_start + i * (bar_height + 10)
        bar_width = int(prob * bar_max_width)
        bar_color = EMOTION_COLORS.get(class_name, (255, 255, 255))
        
        # Draw background
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_max_width, bar_y + bar_height),
            (50, 50, 50),
            -1
        )
        
        # Draw probability bar
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            bar_color,
            -1
        )
        
        # Draw text
        text = f"{class_name}: {prob:.1%}"
        cv2.putText(
            frame,
            text,
            (bar_x + 5, bar_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return frame


def main():
    """
    Main function for real-time emotion detection.
    """
    print("=" * 80)
    print("Real-time Emotion Detection Demo")
    print("=" * 80)
    
    # Check if model checkpoint exists
    checkpoint_path = 'models/fer_resnet18_best.pth'
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Model checkpoint not found: {checkpoint_path}")
        print("\nPlease train the model first:")
        print("  python -m src.training.train_fer_resnet --epochs 20")
        return
    
    # Load emotion predictor
    print("\nLoading emotion predictor...")
    predictor = EmotionPredictor(checkpoint_path)
    
    # Load face detector (Haar Cascade)
    print("Loading face detector...")
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    if face_cascade.empty():
        print(f"Error: Could not load face cascade from {face_cascade_path}")
        return
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n" + "=" * 80)
    print("Starting real-time emotion detection...")
    print("=" * 80)
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("\n")
    
    frame_count = 0
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame_count += 1
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                try:
                    # Predict emotion
                    emotion, label_idx, probabilities = predictor.predict_emotion_from_frame(
                        frame, (x, y, w, h)
                    )
                    
                    # Draw emotion information
                    frame = draw_emotion_info(
                        frame, (x, y, w, h), emotion, probabilities, CLASS_NAMES
                    )
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
            
            # Draw FPS
            if frame_count % 30 == 0:
                cv2.putText(
                    frame,
                    f"Faces: {len(faces)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Draw instructions
            cv2.putText(
                frame,
                "Press 'q' to quit, 's' to save",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Display frame
            cv2.imshow('Emotion Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo completed!")


if __name__ == '__main__':
    main()
