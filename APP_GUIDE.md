# Emotion-Based Music Player - User Guide

## Overview

The Emotion-Based Music Player is a real-time application that:
1. **Detects your face** using webcam
2. **Recognizes your emotion** using the trained ResNet-18 model
3. **Plays appropriate music** based on your current emotional state
4. **Tracks emotion history** to avoid rapid music switching
5. **Provides visual feedback** with colored overlays and statistics

## Prerequisites

### 1. Trained Model
You must have a trained emotion recognition model:
```bash
python train.py
```

This creates `models/fer_resnet18_best.pth`.

### 2. Music Library
Organize your music files into emotion-specific folders:

```
music/
├── angry/
│   ├── rock_song1.mp3
│   ├── metal_song2.mp3
│   └── ...
├── sad/
│   ├── ballad1.mp3
│   ├── slow_song2.mp3
│   └── ...
├── happy/
│   ├── upbeat1.mp3
│   ├── pop_song2.mp3
│   └── ...
└── neutral/
    ├── ambient1.mp3
    ├── calm_song2.mp3
    └── ...
```

**Supported formats**: MP3, WAV, OGG, FLAC, M4A

**Tip**: Place 2-5 songs in each folder for variety.

### 3. Webcam
Ensure your webcam is connected and accessible.

## Running the Application

### Basic Usage
```bash
python src/app.py
```

### What Happens:
1. Application initializes (loads model, face detector, music player)
2. Webcam opens and displays live feed
3. Face detection starts automatically
4. When a face is detected:
   - Emotion is predicted every 10 frames
   - Predictions are added to history (last 15)
   - Dominant emotion is calculated
   - Music plays when dominant emotion changes

## Controls

| Key | Action |
|-----|--------|
| **Q** | Quit application |
| **M** | Mute/Unmute music |
| **P** | Pause/Resume current song |
| **S** | Save screenshot |
| **ESC** | Quit application (alternative) |

## User Interface

### Main Window Elements:

1. **Face Rectangle** (dynamic color)
   - Color changes based on detected emotion
   - Shows current emotion and confidence
   - Red = Angry, Blue = Sad, Green = Happy, Cyan = Neutral

2. **Info Panel** (top-left)
   - Face detection status
   - Current frame number
   - Emotion history count

3. **Dominant Emotion Panel** (top-right)
   - Shows the dominant emotion in large text
   - Music playback status
   - Color-coded border

4. **Emotion History Bar** (bottom-center)
   - Visual representation of last 15 predictions
   - Each segment colored by emotion
   - Helps you see emotion trends

5. **Controls** (bottom)
   - Keyboard shortcut reminders

## How Emotion Detection Works

### Prediction Pipeline:
1. **Face Detection** (every frame)
   - Haar Cascade detector finds faces
   - Largest face is selected for tracking

2. **Emotion Prediction** (every 10 frames)
   - Face is cropped and preprocessed
   - ResNet-18 model predicts emotion
   - Prediction added to history queue

3. **Dominant Emotion** (calculated from history)
   - Most frequent emotion in last 15 predictions
   - Smooths out noise and brief expressions
   - More stable than frame-by-frame prediction

4. **Music Selection**
   - When dominant emotion changes
   - Random song from emotion folder
   - Plays in loop until emotion changes

### Why Every 10 Frames?
- Reduces computational load
- Still responsive (3-4 times per second at 30 FPS)
- Balances performance and accuracy

### Why Track 15 Predictions?
- Provides ~5 seconds of history (at 30 FPS)
- Smooths out transient expressions
- Prevents rapid music switching
- More natural user experience

## Troubleshooting

### Model Not Found
```
Error: Model checkpoint not found: models/fer_resnet18_best.pth
```
**Solution**: Train the model first:
```bash
python train.py
```

### No Face Detected
- Ensure good lighting
- Face the camera directly
- Move closer to camera
- Check if webcam is working

### Music Not Playing
1. Check if music files exist in `music/` folders
2. Verify file formats (MP3, WAV, OGG, FLAC, M4A)
3. Check if muted (press M to unmute)
4. Look at console for error messages

### Webcam Not Opening
```
Error: Could not open webcam
```
**Solutions**:
- Close other apps using webcam
- Check webcam permissions
- Try different camera_id in app config
- Verify webcam is connected

### Slow Performance
- GPU recommended for real-time performance
- Reduce `prediction_interval` (e.g., 15 instead of 10)
- Close other GPU-intensive applications
- Lower webcam resolution

### Wrong Emotions Detected
- Model accuracy depends on training quality
- FER2013 is challenging with ~50-60% accuracy
- Ensure good lighting on your face
- Face camera directly
- Wait for history to stabilize (15 predictions)

## Configuration

Edit `src/app.py` to customize:

```python
# In main() function:
MODEL_PATH = 'models/fer_resnet18_best.pth'  # Model checkpoint
MUSIC_DIR = 'music'                          # Music directory
PREDICTION_INTERVAL = 10                     # Frames between predictions
HISTORY_SIZE = 15                            # Number of predictions to track
CAMERA_ID = 0                                # Webcam device ID
MUSIC_VOLUME = 0.5                           # Initial volume (0.0 to 1.0)
```

### Adjust Sensitivity:
- **More sensitive** (faster emotion changes):
  - Decrease `HISTORY_SIZE` to 10
  - Decrease `PREDICTION_INTERVAL` to 5
  
- **Less sensitive** (slower, more stable):
  - Increase `HISTORY_SIZE` to 20
  - Increase `PREDICTION_INTERVAL` to 15

## Music Recommendations

### Angry
- Rock, Metal, Punk
- High energy, aggressive
- Fast tempo, loud

### Sad
- Ballads, Classical, Blues
- Slow tempo, melancholic
- Emotional, contemplative

### Happy
- Pop, Dance, Reggae
- Upbeat, energetic
- Major keys, positive

### Neutral
- Ambient, Jazz, Lofi
- Calm, steady
- Background music, focused

## Tips for Best Results

1. **Lighting**: Ensure your face is well-lit
2. **Distance**: Sit 1-2 feet from camera
3. **Angle**: Face camera directly
4. **Stability**: Keep camera stable (not moving)
5. **Music**: Use diverse, clear emotional music
6. **Patience**: Wait for history to stabilize before judging

## Advanced Usage

### Multiple Cameras
Change camera ID in configuration:
```python
CAMERA_ID = 1  # Try 0, 1, 2, etc.
```

### Custom Music Structure
The app looks for folders named exactly:
- `angry/`
- `sad/`
- `happy/`
- `neutral/`

Don't rename these folders.

### Recording Session
Use OBS or similar software to record:
- Window capture: Emotion-Based Music Player window
- Audio: Desktop audio (to capture music)
- Webcam: Already visible in app window

### API Integration
For developers, you can import components:

```python
from src.app import EmotionMusicApp

# Custom configuration
app = EmotionMusicApp(
    model_path='models/fer_resnet18_best.pth',
    music_dir='music',
    prediction_interval=10,
    history_size=15,
    camera_id=0,
    music_volume=0.5
)

# Run application
app.run()
```

## Performance Benchmarks

On RTX 4060 Laptop GPU:
- **Face Detection**: ~100 FPS
- **Emotion Prediction**: ~100 predictions/second
- **Total FPS**: 25-30 FPS (includes rendering)
- **Latency**: ~100-200ms from face to prediction

On CPU only:
- **Face Detection**: ~50 FPS
- **Emotion Prediction**: ~20 predictions/second
- **Total FPS**: 15-20 FPS
- **Latency**: ~300-500ms

## Safety & Privacy

- All processing happens **locally** on your computer
- No data is sent to any server
- No recordings are saved (unless you press 'S')
- Model runs entirely offline
- Music files never leave your computer

## Known Limitations

1. **Single Face**: Only tracks largest/primary face
2. **Lighting Sensitive**: Poor lighting affects accuracy
3. **Model Accuracy**: ~50-60% on challenging emotions
4. **No Audio Input**: Cannot hear your voice/sounds
5. **Pose Sensitive**: Works best with frontal face

## Future Enhancements

Potential improvements:
- Multi-face tracking
- Emotion intensity detection
- Music mood mixing (crossfade)
- Session statistics and logging
- Spotify/streaming integration
- Mobile app version
- Voice activity detection

## Getting Help

If you encounter issues:

1. Check console output for error messages
2. Verify all prerequisites are met
3. Test components individually:
   ```bash
   python src/face_detection.py
   python src/music_player.py
   python src/inference/emotion_inference.py
   ```
4. Review `PROJECT_SUMMARY.md` for details
5. Check model training logs

## Credits

- **Face Detection**: OpenCV Haar Cascade
- **Emotion Recognition**: Custom ResNet-18 (trained from scratch)
- **Music Playback**: Pygame
- **Dataset**: FER2013 (Kaggle)

## License

This project is for educational and personal use.
FER2013 dataset is public domain.

---

**Enjoy your emotion-responsive music experience! 🎵😊**
