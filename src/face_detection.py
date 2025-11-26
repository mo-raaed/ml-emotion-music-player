"""
Face Detection Module

This module provides face detection utilities using OpenCV's Haar Cascade
and DNN-based detectors for the emotion recognition application.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """
    Face detector class using OpenCV Haar Cascade.
    
    This detector is fast and suitable for real-time applications.
    """
    
    def __init__(self, method: str = 'haar'):
        """
        Initialize face detector.
        
        Args:
            method: Detection method ('haar' or 'dnn')
        """
        self.method = method
        
        if method == 'haar':
            # Load Haar Cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            
            if self.detector.empty():
                raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")
            
            print(f"Face detector initialized: Haar Cascade")
        
        elif method == 'dnn':
            # Load DNN-based face detector (more accurate but slower)
            model_file = "res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "deploy.prototxt"
            
            try:
                self.detector = cv2.dnn.readNetFromCaffe(config_file, model_file)
                print(f"Face detector initialized: DNN (Caffe)")
            except:
                print("Warning: DNN model not found. Falling back to Haar Cascade.")
                self.method = 'haar'
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.detector = cv2.CascadeClassifier(cascade_path)
        
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def detect_faces(
        self,
        frame: np.ndarray,
        min_size: Tuple[int, int] = (48, 48),
        scale_factor: float = 1.1,
        min_neighbors: int = 5
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Input frame (BGR)
            min_size: Minimum face size (width, height)
            scale_factor: Scale factor for Haar Cascade
            min_neighbors: Minimum neighbors for Haar Cascade
            
        Returns:
            List of face bounding boxes as (x, y, w, h)
        """
        if self.method == 'haar':
            return self._detect_haar(frame, min_size, scale_factor, min_neighbors)
        elif self.method == 'dnn':
            return self._detect_dnn(frame)
        else:
            return []
    
    def _detect_haar(
        self,
        frame: np.ndarray,
        min_size: Tuple[int, int],
        scale_factor: float,
        min_neighbors: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using Haar Cascade.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of tuples
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    
    def _detect_dnn(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using DNN-based detector.
        """
        h, w = frame.shape[:2]
        
        # Prepare blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        # Run detection
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        faces = []
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                # Get bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype(int)
                
                # Convert to (x, y, w, h) format
                x = max(0, x1)
                y = max(0, y1)
                w = x2 - x1
                h = y2 - y1
                
                faces.append((x, y, w, h))
        
        return faces
    
    def get_largest_face(
        self,
        frame: np.ndarray,
        **kwargs
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect and return the largest face in the frame.
        
        This is useful for single-person applications where we want
        to track the main subject.
        
        Args:
            frame: Input frame (BGR)
            **kwargs: Additional arguments for detect_faces
            
        Returns:
            Largest face bounding box as (x, y, w, h) or None if no faces detected
        """
        faces = self.detect_faces(frame, **kwargs)
        
        if len(faces) == 0:
            return None
        
        # Find largest face by area
        largest_face = max(faces, key=lambda bbox: bbox[2] * bbox[3])
        
        return largest_face
    
    def get_center_face(
        self,
        frame: np.ndarray,
        **kwargs
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect and return the face closest to the center of the frame.
        
        Args:
            frame: Input frame (BGR)
            **kwargs: Additional arguments for detect_faces
            
        Returns:
            Center-most face bounding box as (x, y, w, h) or None if no faces detected
        """
        faces = self.detect_faces(frame, **kwargs)
        
        if len(faces) == 0:
            return None
        
        # Get frame center
        frame_h, frame_w = frame.shape[:2]
        frame_center_x = frame_w // 2
        frame_center_y = frame_h // 2
        
        # Find face closest to center
        def distance_from_center(bbox):
            x, y, w, h = bbox
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            dist = np.sqrt(
                (face_center_x - frame_center_x) ** 2 +
                (face_center_y - frame_center_y) ** 2
            )
            return dist
        
        center_face = min(faces, key=distance_from_center)
        
        return center_face


def draw_face_rectangle(
    frame: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
    confidence: Optional[float] = None
) -> np.ndarray:
    """
    Draw a rectangle around a detected face with optional label.
    
    Args:
        frame: Input frame (BGR)
        face_bbox: Face bounding box as (x, y, w, h)
        color: Rectangle color in BGR
        thickness: Rectangle line thickness
        label: Optional label text to display
        confidence: Optional confidence value to display
        
    Returns:
        Frame with drawn rectangle
    """
    x, y, w, h = face_bbox
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label if provided
    if label is not None:
        # Prepare label text
        if confidence is not None:
            text = f"{label}: {confidence:.1%}"
        else:
            text = label
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x, y - text_h - 10),
            (x + text_w + 10, y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (x + 5, y - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return frame


if __name__ == "__main__":
    """
    Test face detection with webcam.
    """
    print("=" * 80)
    print("Face Detection Test")
    print("=" * 80)
    
    # Initialize detector
    detector = FaceDetector(method='haar')
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit(1)
    
    print("\nPress 'q' to quit\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Draw rectangles around all faces
            for face_bbox in faces:
                draw_face_rectangle(frame, face_bbox, label="Face")
            
            # Display info
            cv2.putText(
                frame,
                f"Faces detected: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Show frame
            cv2.imshow('Face Detection Test', frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Test completed!")
