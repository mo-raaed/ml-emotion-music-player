# Project Summary - Facial Emotion Recognition with ResNet-18

## Overview
This project implements a 4-class facial emotion recognition system using PyTorch and ResNet-18 architecture trained from scratch on the FER2013 dataset.

## Emotion Classes
The model classifies faces into 4 emotions:
1. **angry** - Anger and Disgust expressions
2. **sad** - Fear and Sadness expressions  
3. **happy** - Happy expressions
4. **neutral** - Surprise and Neutral expressions

## Architecture
- **Model**: ResNet-18 (random initialization, NO pretrained weights)
- **Input**: 48×48 grayscale images (single channel)
- **Output**: 4-class softmax probabilities
- **Parameters**: ~11.17 million trainable parameters

## Key Modifications
1. **First Conv Layer**: Modified from 3 channels → 1 channel for grayscale input
2. **Final FC Layer**: Modified from 1000 classes → 4 emotion classes
3. **Class Mapping**: FER2013's 7 classes merged into 4 meaningful emotion categories

## Project Structure

```
Machine Learning Project/
├── data/                           # Dataset directory
│   └── fer2013.csv                # FER2013 dataset (place here)
│
├── models/                         # Saved model checkpoints
│   ├── fer_resnet18_best.pth      # Best model checkpoint
│   ├── training_history.png       # Training curves
│   ├── confusion_matrix.png       # Test set confusion matrix
│   └── classification_report.txt  # Detailed metrics
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── fer2013_dataset.py     # Dataset loading and preprocessing
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── fer_resnet.py          # ResNet-18 model definition
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_fer_resnet.py    # Training script
│   │
│   └── inference/
│       ├── __init__.py
│       └── emotion_inference.py   # Inference interface
│
├── train.py                        # Simple training launcher
├── demo_webcam.py                  # Real-time webcam demo
├── demo_image.py                   # Static image demo
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation

```

## Workflow

### 1. Data Loading (`src/data/fer2013_dataset.py`)
- Loads FER2013 CSV file
- Parses pixel strings into 48×48 images
- Applies 7→4 class mapping
- Splits into train/val/test based on Usage column
- Data augmentation for training (flip, rotation, crop)
- Normalizes to [0, 1] range

**Key Classes:**
- `Fer2013Dataset`: PyTorch Dataset for FER2013
- `create_dataloaders()`: Creates train/val/test dataloaders

### 2. Model Definition (`src/models/fer_resnet.py`)
- ResNet-18 with random initialization
- Modified for grayscale 48×48 input
- 4-class output for emotion prediction

**Key Classes:**
- `FerResNet18`: Main model class
- `create_model()`: Model creation with device setup

### 3. Training (`src/training/train_fer_resnet.py`)
- Cross-entropy loss
- Adam optimizer (lr=1e-3)
- StepLR scheduler (reduces LR every 7 epochs)
- Tracks best validation accuracy
- Saves checkpoints with full state
- Evaluates on test set after training
- Generates plots and reports

**Key Functions:**
- `train_one_epoch()`: Single epoch training
- `validate()`: Validation evaluation
- `test_model()`: Final test evaluation
- `save_checkpoint()`: Save model state
- `plot_confusion_matrix()`: Visualization

### 4. Inference (`src/inference/emotion_inference.py`)
- Loads trained checkpoint
- Preprocesses images (crop, resize, normalize)
- Predicts emotion with confidence scores
- GPU-accelerated inference

**Key Classes:**
- `EmotionPredictor`: Main inference class
- `predict_emotion_from_frame()`: Frame-level prediction

## Usage Examples

### Training
```bash
# Simple training
python train.py

# Custom parameters
python -m src.training.train_fer_resnet --epochs 30 --batch-size 128 --lr 0.0005
```

### Inference - Webcam
```bash
python demo_webcam.py
# Press 'q' to quit, 's' to save screenshot
```

### Inference - Image
```bash
python demo_image.py path/to/image.jpg
```

### Inference - Programmatic
```python
from src.inference.emotion_inference import EmotionPredictor
import cv2

# Load model once
predictor = EmotionPredictor('models/fer_resnet18_best.pth')

# Load image
frame = cv2.imread('test.jpg')

# Detect face (x, y, w, h)
face_bbox = (100, 100, 200, 200)

# Predict emotion
emotion, idx, probs = predictor.predict_emotion_from_frame(frame, face_bbox)

print(f"Emotion: {emotion}")
print(f"Confidence: {probs[idx]:.2%}")
```

## Training Configuration

**Default Parameters:**
- Epochs: 20
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Scheduler: StepLR (step=7, gamma=0.1)
- Loss: CrossEntropyLoss

**Data Augmentation (Training Only):**
- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Random crop (90-100%)

**Hardware:**
- GPU: RTX 4060 Laptop GPU
- CUDA: 12.4
- PyTorch: 2.6.0

## Expected Performance

The model is trained from scratch (no pretrained weights), so performance depends on:
- Training epochs (more = better, but risk overfitting)
- Data augmentation effectiveness
- Learning rate schedule
- Class imbalance in FER2013 dataset

Typical metrics after 20 epochs:
- Training accuracy: 60-70%
- Validation accuracy: 50-60%
- Test accuracy: 50-60%

Note: FER2013 is a challenging dataset with noisy labels and class imbalance. State-of-the-art models achieve ~70-75% on the original 7 classes.

## Class Distribution

After 7→4 mapping, the dataset is imbalanced:
- **happy**: ~35-40% (largest class)
- **neutral**: ~25-30%
- **sad**: ~20-25%
- **angry**: ~15-20% (smallest class)

This imbalance may affect model performance, particularly for the `angry` class.

## Output Files

After training, the `models/` directory contains:

1. **fer_resnet18_best.pth** - Best model checkpoint
   - Model state dict
   - Optimizer state dict
   - Epoch number
   - Validation accuracy
   - Class names

2. **training_history.png** - Training curves
   - Loss curves (train vs val)
   - Accuracy curves (train vs val)

3. **confusion_matrix.png** - Test set confusion matrix
   - Shows prediction vs true label distribution

4. **classification_report.txt** - Detailed metrics
   - Precision, recall, F1-score per class
   - Support (number of samples) per class
   - Overall accuracy

## Troubleshooting

**Issue: CUDA out of memory**
- Reduce batch size: `--batch-size 32`
- Use CPU: Set `use_gpu_if_available=False`

**Issue: Low accuracy**
- Train for more epochs: `--epochs 30`
- Try different learning rate: `--lr 0.0005`
- Check class distribution and consider weighted loss

**Issue: Overfitting (train acc >> val acc)**
- Increase data augmentation
- Add dropout or other regularization
- Reduce model capacity
- Use early stopping

**Issue: Slow inference**
- Ensure GPU is being used
- Batch multiple faces together
- Use model.eval() and torch.no_grad()

## Future Improvements

1. **Model Architecture:**
   - Try deeper networks (ResNet-34, ResNet-50)
   - Experiment with attention mechanisms
   - Use ensemble of multiple models

2. **Data:**
   - Balance classes with weighted sampling
   - Use external datasets for pretraining
   - Clean noisy labels manually

3. **Training:**
   - Implement focal loss for class imbalance
   - Use learning rate warmup
   - Try different optimizers (AdamW, SGD with momentum)

4. **Inference:**
   - Add temporal smoothing for video
   - Implement face tracking for stable predictions
   - Optimize model with TorchScript or ONNX

5. **Application:**
   - Add emotion-based music recommendation
   - Create web interface with Flask/FastAPI
   - Deploy as mobile app with TensorFlow Lite

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- PyTorch 2.0+
- torchvision
- OpenCV
- NumPy, Pandas
- scikit-learn
- matplotlib, seaborn

## License & Dataset

**FER2013 Dataset:**
- Citation: "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, et al. 2013
- Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
- License: Public domain (no restrictions)

## Contact & Credits

This project implements a standard ResNet-18 architecture for emotion recognition, using PyTorch best practices and clean code structure suitable for research and production deployment.

---

**Project Status**: ✅ Complete and ready for training and inference
**Last Updated**: November 26, 2025
