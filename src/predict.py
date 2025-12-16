"""
Prediction module for instrument recognition.
Load trained models and make predictions on new audio files.
"""

import numpy as np
import librosa
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import json


class InstrumentPredictor:
    """Make predictions on audio files using trained models."""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config.yaml"
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model file
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model and preprocessing objects
        self.model = joblib.load(model_path)
        
        model_dir = Path(model_path).parent
        self.scaler = joblib.load(model_dir / 'scaler.pkl')
        self.label_encoder = joblib.load(model_dir / 'label_encoder.pkl')
        
        # Audio processing parameters
        self.sample_rate = self.config['audio']['sample_rate']
        self.duration = self.config['audio']['duration']
        self.offset = self.config['audio']['offset']
        
        # Feature extraction parameters
        self.n_mfcc = self.config['features']['n_mfcc']
        self.n_fft = self.config['features']['n_fft']
        self.hop_length = self.config['features']['hop_length']
        self.n_mels = self.config['features']['n_mels']
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio signal as numpy array
        """
        audio, sr = librosa.load(
            file_path,
            sr=self.sample_rate,
            duration=self.duration,
            offset=self.offset
        )
        return audio
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract features from audio signal.
        
        Args:
            audio: Audio signal
            
        Returns:
            Feature vector
        """
        # MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta = np.mean(librosa.feature.delta(mfcc), axis=1)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y=audio,
            hop_length=self.hop_length
        )
        
        spectral_features = np.array([
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate)
        ])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # Combine all features
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            mfcc_delta,
            spectral_features,
            chroma_mean,
            chroma_std
        ])
        
        return features
    
    def predict(self, file_path: str) -> Dict:
        """
        Predict instrument for a single audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary containing prediction results
        """
        # Load and extract features
        audio = self.load_audio(file_path)
        features = self.extract_features(audio)
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction_encoded = self.model.predict(features_scaled)[0]
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            prob_dict = {
                label: float(prob)
                for label, prob in zip(self.label_encoder.classes_, probabilities)
            }
        else:
            prob_dict = None
        
        result = {
            'file': file_path,
            'prediction': prediction,
            'probabilities': prob_dict
        }
        
        return result
    
    def predict_batch(self, file_paths: List[str]) -> List[Dict]:
        """
        Predict instruments for multiple audio files.
        
        Args:
            file_paths: List of audio file paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.predict(file_path)
                results.append(result)
                print(f"✓ {Path(file_path).name}: {result['prediction']}")
            except Exception as e:
                print(f"✗ {Path(file_path).name}: Error - {e}")
                results.append({
                    'file': file_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_top_k(self, file_path: str, k: int = 3) -> Dict:
        """
        Get top-k predictions for an audio file.
        
        Args:
            file_path: Path to audio file
            k: Number of top predictions to return
            
        Returns:
            Dictionary with top-k predictions
        """
        result = self.predict(file_path)
        
        if result.get('probabilities'):
            # Sort by probability
            sorted_probs = sorted(
                result['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_k_predictions = [
                {'instrument': label, 'probability': prob}
                for label, prob in sorted_probs[:k]
            ]
            
            result['top_k_predictions'] = top_k_predictions
        
        return result


class ModelComparator:
    """Compare predictions from multiple models."""
    
    def __init__(self, model_dir: str, config_path: str = "config.yaml"):
        """
        Initialize comparator.
        
        Args:
            model_dir: Directory containing trained models
            config_path: Path to configuration file
        """
        self.model_dir = Path(model_dir)
        self.config_path = config_path
        self.predictors = {}
        
        # Load all models
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all trained models from directory."""
        model_files = list(self.model_dir.glob('*_model.pkl'))
        
        for model_file in model_files:
            model_name = model_file.stem.replace('_model', '')
            self.predictors[model_name] = InstrumentPredictor(
                str(model_file),
                self.config_path
            )
            print(f"Loaded model: {model_name}")
    
    def compare_predictions(self, file_path: str) -> Dict:
        """
        Compare predictions from all models for a single file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with predictions from all models
        """
        results = {
            'file': file_path,
            'predictions': {}
        }
        
        for model_name, predictor in self.predictors.items():
            try:
                prediction = predictor.predict(file_path)
                results['predictions'][model_name] = prediction
            except Exception as e:
                results['predictions'][model_name] = {'error': str(e)}
        
        return results
    
    def evaluate_consensus(self, file_path: str) -> str:
        """
        Get consensus prediction from all models.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Consensus prediction
        """
        predictions = []
        
        for predictor in self.predictors.values():
            try:
                result = predictor.predict(file_path)
                predictions.append(result['prediction'])
            except:
                continue
        
        # Majority voting
        if predictions:
            consensus = max(set(predictions), key=predictions.count)
            return consensus
        
        return "Unknown"


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file_path> [model_name]")
        print("Available models: svm, random_forest, xgboost")
        print("\nExample:")
        print("  python predict.py audio.wav svm")
        print("  python predict.py audio.wav  # Uses all models")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_dir = "models"
    
    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        # Use specific model
        model_name = sys.argv[2]
        model_path = f"{model_dir}/{model_name}_model.pkl"
        
        if not Path(model_path).exists():
            print(f"Error: Model not found: {model_path}")
            sys.exit(1)
        
        predictor = InstrumentPredictor(model_path)
        result = predictor.predict_top_k(audio_file, k=3)
        
        print(f"\nPrediction for: {audio_file}")
        print(f"Predicted instrument: {result['prediction']}")
        
        if result.get('top_k_predictions'):
            print("\nTop 3 predictions:")
            for i, pred in enumerate(result['top_k_predictions'], 1):
                print(f"{i}. {pred['instrument']}: {pred['probability']:.4f}")
    else:
        # Compare all models
        comparator = ModelComparator(model_dir)
        results = comparator.compare_predictions(audio_file)
        
        print(f"\nPredictions for: {audio_file}")
        print("-" * 50)
        
        for model_name, prediction in results['predictions'].items():
            if 'error' in prediction:
                print(f"{model_name}: Error - {prediction['error']}")
            else:
                print(f"{model_name}: {prediction['prediction']}")
        
        consensus = comparator.evaluate_consensus(audio_file)
        print(f"\nConsensus prediction: {consensus}")
