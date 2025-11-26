"""
Emotion-Based Music Player Application

This application combines face detection, emotion recognition, and music playback
to create a real-time emotion-responsive music player.

The app:
1. Captures video from webcam
2. Detects faces in each frame
3. Predicts emotions using trained ResNet-18 model
4. Tracks emotion history and determines dominant emotion
5. Plays emotion-appropriate music from local library
6. Displays real-time visualization

Usage:
    python src/app.py
    
Controls:
    - Press 'q' to quit
    - Press 'm' to mute/unmute
    - Press 's' to save screenshot
    - Press 'p' to pause/resume music
"""

import cv2
import numpy as np
import os
from collections import Counter, deque
from datetime import datetime
from typing import Optional, Tuple

from src.inference.emotion_inference import EmotionPredictor
from src.face_detection import FaceDetector, draw_face_rectangle
from src.music_player import EmotionMusicPlayer
from src.data.fer2013_dataset import CLASS_NAMES


# Colors for each emotion (BGR format)
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'sad': (255, 0, 0),        # Blue
    'happy': (0, 255, 0),      # Green
    'neutral': (255, 255, 0),  # Cyan
}


class EmotionMusicApp:
    """
    Main application class for emotion-based music player.
    """
    
    def __init__(
        self,
        model_path: str = 'models/fer_resnet18_best.pth',
        music_dir: str = 'music',
        prediction_interval: int = 10,
        history_size: int = 15,
        camera_id: int = 0,
        music_volume: float = 0.5
    ):
        """
        Initialize the application.
        
        Args:
            model_path: Path to trained emotion model
            music_dir: Directory containing emotion-specific music
            prediction_interval: Predict emotion every N frames
            history_size: Number of recent predictions to track
            camera_id: Camera device ID
            music_volume: Initial music volume (0.0 to 1.0)
        """
        print("=" * 80)
        print("Emotion-Based Music Player")
        print("=" * 80)
        
        # Configuration
        self.prediction_interval = prediction_interval
        self.history_size = history_size
        self.camera_id = camera_id
        
        # State
        self.frame_count = 0
        self.emotion_history = deque(maxlen=history_size)
        self.dominant_emotion = None
        self.last_prediction = None
        self.current_face_bbox = None
        self.is_muted = False
        
        # Initialize components
        print("\nInitializing components...")
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"\nError: Model checkpoint not found: {model_path}")
            print("\nPlease train the model first:")
            print("  python train.py")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load emotion predictor
        print("\n1. Loading emotion recognition model...")
        self.emotion_predictor = EmotionPredictor(model_path)
        
        # Initialize face detector
        print("\n2. Initializing face detector...")
        self.face_detector = FaceDetector(method='haar')
        
        # Initialize music player
        print("\n3. Initializing music player...")
        self.music_player = EmotionMusicPlayer(music_dir, volume=music_volume)
        
        # Open webcam
        print("\n4. Opening webcam...")
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "=" * 80)
        print("Initialization complete!")
        print("=" * 80)
    
    def get_dominant_emotion(self) -> Optional[str]:
        """
        Compute the dominant (most frequent) emotion from history.
        
        Returns:
            Dominant emotion label or None if history is empty
        """
        if len(self.emotion_history) == 0:
            return None
        
        # Count occurrences
        emotion_counts = Counter(self.emotion_history)
        
        # Get most common emotion
        dominant = emotion_counts.most_common(1)[0][0]
        
        return dominant
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect face, predict emotion, update UI.
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            Processed frame with overlays
        """
        self.frame_count += 1
        
        # Detect main face (largest or center-most)
        face_bbox = self.face_detector.get_largest_face(frame)
        
        if face_bbox is not None:
            self.current_face_bbox = face_bbox
            
            # Predict emotion every N frames
            if self.frame_count % self.prediction_interval == 0:
                try:
                    # Predict emotion
                    emotion, label_idx, probabilities = self.emotion_predictor.predict_emotion_from_frame(
                        frame, face_bbox
                    )
                    
                    # Store prediction
                    self.last_prediction = (emotion, probabilities)
                    
                    # Add to history
                    self.emotion_history.append(emotion)
                    
                    # Check for dominant emotion change
                    new_dominant = self.get_dominant_emotion()
                    
                    if new_dominant != self.dominant_emotion:
                        self.dominant_emotion = new_dominant
                        print(f"\n{'='*60}")
                        print(f"Emotion changed: {self.dominant_emotion.upper()}")
                        print(f"{'='*60}")
                        
                        # Play corresponding music
                        if not self.is_muted:
                            self.music_player.play_emotion(self.dominant_emotion)
                
                except Exception as e:
                    print(f"Error predicting emotion: {e}")
        
        else:
            # No face detected
            self.current_face_bbox = None
        
        # Draw UI elements
        frame = self.draw_ui(frame)
        
        return frame
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw UI elements on the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with UI overlays
        """
        # Draw face rectangle if detected
        if self.current_face_bbox is not None and self.last_prediction is not None:
            emotion, probabilities = self.last_prediction
            confidence = probabilities[CLASS_NAMES.index(emotion)]
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            
            draw_face_rectangle(
                frame,
                self.current_face_bbox,
                color=color,
                thickness=3,
                label=f"Current: {emotion}",
                confidence=confidence
            )
        
        # Draw dominant emotion panel
        if self.dominant_emotion is not None:
            self.draw_emotion_panel(frame)
        
        # Draw emotion history bar
        if len(self.emotion_history) > 0:
            self.draw_history_bar(frame)
        
        # Draw info panel
        self.draw_info_panel(frame)
        
        # Draw controls
        self.draw_controls(frame)
        
        return frame
    
    def draw_emotion_panel(self, frame: np.ndarray):
        """Draw the dominant emotion panel."""
        h, w = frame.shape[:2]
        
        # Panel properties
        panel_width = 250
        panel_height = 120
        panel_x = w - panel_width - 20
        panel_y = 20
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw border
        color = EMOTION_COLORS.get(self.dominant_emotion, (255, 255, 255))
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            color,
            3
        )
        
        # Draw title
        cv2.putText(
            frame,
            "Dominant Emotion",
            (panel_x + 10, panel_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        # Draw emotion
        cv2.putText(
            frame,
            self.dominant_emotion.upper(),
            (panel_x + 10, panel_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            2
        )
        
        # Draw music status
        status = self.music_player.get_status()
        music_text = "♪ Playing" if status['is_playing'] else "♪ No music"
        if self.is_muted:
            music_text = "🔇 Muted"
        
        cv2.putText(
            frame,
            music_text,
            (panel_x + 10, panel_y + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    def draw_history_bar(self, frame: np.ndarray):
        """Draw emotion history as a colored bar."""
        h, w = frame.shape[:2]
        
        bar_width = 300
        bar_height = 30
        bar_x = (w - bar_width) // 2
        bar_y = h - 100
        
        # Draw background
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50),
            -1
        )
        
        # Draw history segments
        segment_width = bar_width / self.history_size
        
        for i, emotion in enumerate(self.emotion_history):
            x_start = int(bar_x + i * segment_width)
            x_end = int(bar_x + (i + 1) * segment_width)
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            
            cv2.rectangle(
                frame,
                (x_start, bar_y),
                (x_end, bar_y + bar_height),
                color,
                -1
            )
        
        # Draw border
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (255, 255, 255),
            2
        )
        
        # Draw label
        cv2.putText(
            frame,
            "Emotion History",
            (bar_x, bar_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    def draw_info_panel(self, frame: np.ndarray):
        """Draw info panel with frame count and face detection status."""
        # Face detection status
        face_status = "Face: Detected" if self.current_face_bbox else "Face: Not detected"
        color = (0, 255, 0) if self.current_face_bbox else (0, 0, 255)
        
        cv2.putText(
            frame,
            face_status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        # Frame count
        cv2.putText(
            frame,
            f"Frame: {self.frame_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # History size
        cv2.putText(
            frame,
            f"History: {len(self.emotion_history)}/{self.history_size}",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    def draw_controls(self, frame: np.ndarray):
        """Draw control instructions at the bottom."""
        h = frame.shape[0]
        
        controls = "Q: Quit | M: Mute/Unmute | P: Pause/Resume | S: Screenshot"
        
        # Draw background
        cv2.rectangle(
            frame,
            (0, h - 40),
            (frame.shape[1], h),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            controls,
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    def handle_key(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            False if should quit, True otherwise
        """
        if key == ord('q') or key == 27:  # 'q' or ESC
            return False
        
        elif key == ord('m'):  # Mute/unmute
            self.is_muted = not self.is_muted
            if self.is_muted:
                self.music_player.stop()
                print("🔇 Muted")
            else:
                if self.dominant_emotion:
                    self.music_player.play_emotion(self.dominant_emotion)
                print("🔊 Unmuted")
        
        elif key == ord('p'):  # Pause/resume
            if self.music_player.is_playing:
                self.music_player.pause()
            else:
                self.music_player.resume()
        
        elif key == ord('s'):  # Screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            frame = self.cap.read()[1]
            if frame is not None:
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        return True
    
    def run(self):
        """
        Main application loop.
        """
        print("\n" + "=" * 80)
        print("Starting Emotion-Based Music Player")
        print("=" * 80)
        print("\nControls:")
        print("  Q - Quit")
        print("  M - Mute/Unmute music")
        print("  P - Pause/Resume music")
        print("  S - Save screenshot")
        print("\nWaiting for face detection...")
        print()
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Emotion-Based Music Player', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key):
                    break
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n" + "=" * 80)
        print("Shutting down...")
        print("=" * 80)
        
        print("Stopping music...")
        self.music_player.cleanup()
        
        print("Releasing webcam...")
        self.cap.release()
        
        print("Closing windows...")
        cv2.destroyAllWindows()
        
        print("\nGoodbye!")


def main():
    """
    Main entry point.
    """
    # Configuration
    MODEL_PATH = 'models/fer_resnet18_best.pth'
    MUSIC_DIR = 'music'
    PREDICTION_INTERVAL = 10  # Predict every 10 frames
    HISTORY_SIZE = 15         # Track last 15 predictions
    CAMERA_ID = 0             # Default webcam
    MUSIC_VOLUME = 0.5        # 50% volume
    
    try:
        # Create and run app
        app = EmotionMusicApp(
            model_path=MODEL_PATH,
            music_dir=MUSIC_DIR,
            prediction_interval=PREDICTION_INTERVAL,
            history_size=HISTORY_SIZE,
            camera_id=CAMERA_ID,
            music_volume=MUSIC_VOLUME
        )
        
        app.run()
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("  1. Model is trained: python train.py")
        print("  2. Music files are in: music/angry/, music/sad/, music/happy/, music/neutral/")
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
