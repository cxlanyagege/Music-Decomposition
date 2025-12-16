# Music Decomposition

A machine learning system for identifying musical instruments in audio files using traditional ML algorithms (SVM, Random Forest, XGBoost) with MFCC and spectral features.

## Features

- **Multiple Algorithms**: Support for SVM, Random Forest, and XGBoost classifiers
- **Rich Feature Extraction**: MFCC, spectral features, and chroma features
- **Complete Pipeline**: Data loading, preprocessing, training, and inference
- **Model Comparison**: Compare predictions across different models
- **Comprehensive Evaluation**: Confusion matrices, classification reports, and metrics

## Project Structure

```
Music Decomposition/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md
├── src/
│   ├── data_loader.py          # Data loading and splitting
│   ├── feature_extractor.py    # Audio feature extraction
│   ├── train.py                # Model training
│   ├── predict.py              # Inference and prediction
│   └── utils.py                # Utility functions
├── data/
│   ├── raw/                    # Raw audio files
│   │   ├── instrument1/
│   │   │   ├── audio1.wav
│   │   │   └── audio2.wav
│   │   └── instrument2/
│   └── processed/              # Processed features
├── models/                     # Trained models
└── results/                    # Evaluation results
```

## Installation

### Prerequisites

- Python 3.14 or higher
- FFmpeg (for audio processing)

### Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Dataset

Organize your audio files in the following structure:

```
data/raw/
├── guitar/
│   ├── guitar_01.wav
│   ├── guitar_02.wav
│   └── ...
├── piano/
│   ├── piano_01.wav
│   └── ...
├── drums/
└── violin/
```

Supported formats: WAV, MP3, FLAC, OGG

### 2. Configure Settings

Edit `config.yaml` to adjust:
- Audio processing parameters (sample rate, duration)
- Feature extraction settings (MFCC coefficients, FFT size)
- Model hyperparameters
- Train/validation/test split ratios

### 3. Load and Split Data

```bash
python src/data_loader.py
```

This will:
- Scan the `data/raw/` directory
- Split data into train/val/test sets (default: 70/15/15)
- Save split information to `data/processed/splits.json`

### 4. Extract Features

```bash
python src/feature_extractor.py
```

This will:
- Extract MFCC, spectral, and chroma features from audio files
- Save features to `data/processed/{train,val,test}_features.npz`

**Extracted features:**
- **MFCC**: 20 coefficients + deltas (60 features)
- **Spectral**: Centroid, rolloff, zero-crossing rate (6 features)
- **Chroma**: 12 pitch classes (24 features)
- **Total**: 90 features per audio sample

### 5. Train Models

```bash
python src/train.py
```

This will:
- Train SVM, Random Forest, and XGBoost models
- Evaluate on train, validation, and test sets
- Generate confusion matrices
- Save trained models to `models/`
- Save evaluation results to `results/evaluation_results.json`

### 6. Make Predictions

**Single model prediction:**
```bash
python src/predict.py path/to/audio.wav svm
```

**Compare all models:**
```bash
python src/predict.py path/to/audio.wav
```

## Usage Examples

### Python API

**Load and predict with a single model:**

```python
from src.predict import InstrumentPredictor

# Initialize predictor with trained model
predictor = InstrumentPredictor("models/xgboost_model.pkl")

# Predict instrument
result = predictor.predict("audio.wav")
print(f"Predicted: {result['prediction']}")

# Get top-3 predictions with probabilities
result = predictor.predict_top_k("audio.wav", k=3)
for pred in result['top_k_predictions']:
    print(f"{pred['instrument']}: {pred['probability']:.2%}")
```

**Compare multiple models:**

```python
from src.predict import ModelComparator

# Initialize comparator
comparator = ModelComparator("models/")

# Compare predictions from all models
results = comparator.compare_predictions("audio.wav")

# Get consensus prediction
consensus = comparator.evaluate_consensus("audio.wav")
print(f"Consensus: {consensus}")
```

**Custom feature extraction:**

```python
from src.feature_extractor import AudioFeatureExtractor

extractor = AudioFeatureExtractor()

# Extract features from a file
features = extractor.process_audio_file("audio.wav")

# Get feature names
feature_names = extractor.get_feature_names()
print(f"Total features: {len(feature_names)}")
```

## Configuration

### Audio Processing Parameters

```yaml
audio:
  sample_rate: 22050      # Sampling rate (Hz)
  duration: 3.0           # Audio duration to process (seconds)
  offset: 0.5             # Skip first N seconds
```

### Feature Extraction

```yaml
features:
  n_mfcc: 20             # Number of MFCC coefficients
  n_fft: 2048            # FFT window size
  hop_length: 512        # Hop length for STFT
  n_mels: 128            # Number of Mel bands
```

### Model Hyperparameters

**SVM:**
```yaml
svm:
  kernel: "rbf"          # Kernel type
  C: 1.0                 # Regularization parameter
  gamma: "scale"         # Kernel coefficient
```

**Random Forest:**
```yaml
random_forest:
  n_estimators: 100      # Number of trees
  max_depth: 20          # Maximum tree depth
  min_samples_split: 2   # Minimum samples to split
```

**XGBoost:**
```yaml
xgboost:
  n_estimators: 100      # Number of boosting rounds
  max_depth: 6           # Maximum tree depth
  learning_rate: 0.1     # Learning rate
```

## Recommended Datasets

### Public Datasets

1. **IRMAS Dataset** (Instrument Recognition in Musical Audio Signals)
   - 11 instrument classes
   - ~6,700 audio excerpts
   - [Download](https://www.upf.edu/web/mtg/irmas)

2. **NSynth Dataset** (Google Magenta)
   - 300,000+ musical notes
   - 1,000 instruments
   - [Download](https://magenta.tensorflow.org/datasets/nsynth)

3. **OpenMIC-2018**
   - 20,000 audio files
   - 20 instrument classes
   - [Download](https://github.com/cosmir/openmic-2018)

4. **MedleyDB**
   - Multi-track recordings
   - Instrument annotations
   - [Download](https://medleydb.weebly.com/)

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and weighted average
- **Recall**: Per-class and weighted average
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions
- **Classification Report**: Detailed per-class metrics

Results are saved in `results/evaluation_results.json` with the following structure:

```json
{
  "model_name": {
    "train": {...},
    "val": {...},
    "test": {
      "accuracy": 0.85,
      "precision": 0.84,
      "recall": 0.85,
      "f1_score": 0.84,
      "confusion_matrix": [[...]],
      "classification_report": {...}
    }
  }
}
```

## Performance Tips

### For Better Accuracy

1. **Data Quality**: Use high-quality audio recordings
2. **Data Augmentation**: Add pitch shifting, time stretching
3. **Feature Engineering**: Experiment with different feature combinations
4. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV
5. **Ensemble Methods**: Combine predictions from multiple models

### For Faster Training

1. **Reduce Data**: Use smaller audio duration
2. **Feature Selection**: Remove low-importance features
3. **Parallel Processing**: Enable `n_jobs=-1` in models
4. **GPU Acceleration**: Use XGBoost with GPU support

## Troubleshooting

### Common Issues

**1. FFmpeg not found:**
```bash
# Install FFmpeg first
brew install ffmpeg  # macOS
sudo apt-get install ffmpeg  # Linux
```

**2. Memory errors during training:**
- Reduce batch size or audio duration
- Use fewer features
- Process data in chunks

**3. Low accuracy:**
- Check data quality and balance
- Increase audio duration
- Tune hyperparameters
- Add more training data

**4. Audio loading errors:**
- Ensure audio files are not corrupted
- Check file format compatibility
- Verify sample rate settings

## Advanced Usage

### Custom Model Training

```python
from src.train import InstrumentClassifier
import numpy as np

# Initialize classifier
classifier = InstrumentClassifier()

# Load your features
X_train, y_train = classifier.load_features("data/processed/train_features.npz")
X_val, y_val = classifier.load_features("data/processed/val_features.npz")
X_test, y_test = classifier.load_features("data/processed/test_features.npz")

# Prepare data
X_train_prep, y_train_prep, X_val_prep, y_val_prep = classifier.prepare_data(
    X_train, y_train, X_val, y_val
)

# Train specific model
model = classifier.train_xgboost(X_train_prep, y_train_prep)

# Evaluate
X_test_prep = classifier.scaler.transform(X_test)
y_test_prep = classifier.label_encoder.transform(y_test)
results = classifier.evaluate_model(model, X_test_prep, y_test_prep, 'xgboost', 'test')
```

## Future Enhancements

Potential improvements for the project:

- [ ] Deep learning models (CNN, RNN, Transformer)
- [ ] Multi-label classification for mixed instruments
- [ ] Real-time audio stream processing
- [ ] Web API for predictions
- [ ] Data augmentation pipeline
- [ ] Cross-validation support
- [ ] Feature importance visualization
- [ ] Hyperparameter optimization

## License

TBA

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Librosa](https://librosa.org/) for audio processing
- [scikit-learn](https://scikit-learn.org/) for ML algorithms
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting
- IRMAS and NSynth datasets for research data

## Contact

For questions or issues, please open an issue on GitHub.
