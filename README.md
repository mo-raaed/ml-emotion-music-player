# Facial Emotion Recognition Project

## Project Structure

```
.
├── data/                    # Raw and processed FER2013 dataset
├── models/                  # Saved model checkpoints (.pth files)
├── music/                   # Music files for emotion-based recommendations
├── notebooks/               # Jupyter notebooks for experimentation
├── src/
│   ├── data/               # Dataset loading and preprocessing
│   ├── models/             # Model architecture definitions
│   ├── training/           # Training loops and utilities
│   └── inference/          # Inference and prediction scripts
└── requirements.txt        # Python dependencies

```

## Installation

Install all dependencies directly to your global Python environment:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python Pillow tqdm tensorboard albumentations
```

## Project Overview

- **Architecture**: ResNet-18 (random initialization, trained from scratch)
- **Dataset**: FER2013 (7 classes → 4 emotion classes)
- **Hardware**: RTX 4060
- **Framework**: PyTorch

## Emotion Classes

The model maps FER2013's 7 original classes to 4 target emotions:
- **angry**: Angry, Disgust
- **sad**: Fear, Sad
- **happy**: Happy
- **neutral**: Surprise, Neutral

## Key Features

- Train emotion recognition from scratch (no pretrained weights)
- 4-class emotion classification with custom mapping
- Real-time inference capabilities
- Emotion-based music recommendation integration
- Comprehensive training metrics and visualization
- Automatic GPU utilization (RTX 4060)

## Dataset Setup

1. Download the FER2013 dataset (CSV format)
2. Place `fer2013.csv` in the `data/` directory

Download from:
- Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

## Training

### Option 1: Simple Training (Recommended)
```bash
python train.py
```

### Option 2: Custom Parameters
```bash
python -m src.training.train_fer_resnet --epochs 20 --batch-size 64 --lr 0.001
```

### Training Arguments
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--csv-path`: Path to fer2013.csv (default: data/fer2013.csv)
- `--checkpoint-dir`: Directory for saving models (default: models)
- `--num-workers`: Data loading workers (default: 4)

### Training Output

The script will:
- Train the model on FER2013 with the 4-class mapping
- Display per-epoch training and validation metrics
- Save the best model to `models/fer_resnet18_best.pth`
- Generate training history plots
- Evaluate on test set with confusion matrix and classification report
- Save all results to the `models/` directory

## Inference

### Real-time Webcam Demo
```bash
python demo_webcam.py
```
Controls:
- Press `q` to quit
- Press `s` to save screenshot

### Static Image Inference
```bash
python demo_image.py path/to/image.jpg
```

### Programmatic Usage
```python
from src.inference.emotion_inference import EmotionPredictor

# Initialize predictor (do this once)
predictor = EmotionPredictor('models/fer_resnet18_best.pth')

# Predict emotion from frame with face bounding box
emotion, label_idx, probabilities = predictor.predict_emotion_from_frame(
    frame_bgr,  # OpenCV BGR image
    (x, y, w, h)  # Face bounding box
)

print(f"Emotion: {emotion}")
print(f"Confidence: {probabilities[label_idx]:.2%}")
```

### Inference Features
- GPU-accelerated prediction (automatic fallback to CPU)
- Real-time performance (suitable for webcam/video)
- Face detection using OpenCV Haar Cascade
- Color-coded emotion visualization
- Confidence scores for all emotion classes

## Emotion-Based Music Player App

Run the complete emotion-responsive music player:

```bash
python src/app.py
```

### Features:
- **Real-time face detection** and emotion recognition
- **Emotion tracking** with history-based smoothing
- **Automatic music playback** based on detected emotion
- **Visual feedback** with colored overlays and emotion history
- **Interactive controls** (mute, pause, screenshot)

### Music Setup:
Place your music files in emotion-specific folders:
```
music/
├── angry/    - Intense, aggressive music
├── sad/      - Melancholic, slow music
├── happy/    - Upbeat, energetic music
└── neutral/  - Calm, ambient music
```

Supported formats: MP3, WAV, OGG, FLAC, M4A

### Controls:
- **Q**: Quit application
- **M**: Mute/unmute music
- **P**: Pause/resume music
- **S**: Save screenshot

### How It Works:
1. Detects your face in real-time using webcam
2. Predicts emotion every 10 frames using trained ResNet-18
3. Tracks last 15 predictions to determine dominant emotion
4. Automatically plays emotion-appropriate music when emotion changes
5. Displays visual feedback with colored overlays and history bar
