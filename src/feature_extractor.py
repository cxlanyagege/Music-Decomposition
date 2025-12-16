"""
Feature extraction module for audio signals.
Extracts MFCC and other audio features for instrument recognition.
"""

import librosa
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import joblib


class AudioFeatureExtractor:
    """Extract audio features from audio signals."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the feature extractor.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sample_rate = self.config['audio']['sample_rate']
        self.n_mfcc = self.config['features']['n_mfcc']
        self.n_fft = self.config['features']['n_fft']
        self.hop_length = self.config['features']['hop_length']
        self.n_mels = self.config['features']['n_mels']
    
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio signal.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            MFCC features (flattened)
        """
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Calculate statistics over time
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta = np.mean(librosa.feature.delta(mfcc), axis=1)
        
        # Concatenate features
        features = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta])
        
        return features
    
    def extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral features from audio signal.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Spectral features
        """
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y=audio,
            hop_length=self.hop_length
        )
        
        # Calculate statistics
        features = np.array([
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate)
        ])
        
        return features
    
    def extract_chroma_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract chroma features from audio signal.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Chroma features
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Calculate statistics
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        features = np.concatenate([chroma_mean, chroma_std])
        
        return features
    
    def extract_all_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract all features from audio signal.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Combined feature vector
        """
        mfcc_features = self.extract_mfcc_features(audio)
        spectral_features = self.extract_spectral_features(audio)
        chroma_features = self.extract_chroma_features(audio)
        
        # Combine all features
        all_features = np.concatenate([
            mfcc_features,
            spectral_features,
            chroma_features
        ])
        
        return all_features
    
    def process_audio_file(self, file_path: str) -> np.ndarray:
        """
        Load audio file and extract features.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Feature vector or None if error
        """
        try:
            # Load audio
            audio, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                duration=self.config['audio']['duration'],
                offset=self.config['audio']['offset']
            )
            
            # Extract features
            features = self.extract_all_features(audio)
            
            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def process_dataset(
        self,
        file_paths: List[str],
        labels: List[str],
        output_path: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a list of audio files and extract features.
        
        Args:
            file_paths: List of audio file paths
            labels: List of corresponding labels
            output_path: Optional path to save features
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        features_list = []
        valid_labels = []
        
        print(f"Processing {len(file_paths)} audio files...")
        
        for file_path, label in tqdm(zip(file_paths, labels), total=len(file_paths)):
            features = self.process_audio_file(file_path)
            
            if features is not None:
                features_list.append(features)
                valid_labels.append(label)
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(valid_labels)
        
        print(f"Extracted features shape: {X.shape}")
        print(f"Valid samples: {len(valid_labels)}/{len(file_paths)}")
        
        # Save if output path provided
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            np.savez(output_path, features=X, labels=y)
            print(f"Features saved to {output_path}")
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all extracted features.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        # MFCC features
        for i in range(self.n_mfcc):
            feature_names.append(f"mfcc_{i}_mean")
        for i in range(self.n_mfcc):
            feature_names.append(f"mfcc_{i}_std")
        for i in range(self.n_mfcc):
            feature_names.append(f"mfcc_{i}_delta")
        
        # Spectral features
        feature_names.extend([
            'spectral_centroid_mean',
            'spectral_centroid_std',
            'spectral_rolloff_mean',
            'spectral_rolloff_std',
            'zero_crossing_rate_mean',
            'zero_crossing_rate_std'
        ])
        
        # Chroma features
        for i in range(12):
            feature_names.append(f"chroma_{i}_mean")
        for i in range(12):
            feature_names.append(f"chroma_{i}_std")
        
        return feature_names


if __name__ == "__main__":
    # Example usage
    extractor = AudioFeatureExtractor()
    
    # Load splits
    splits_path = "data/processed/splits.json"
    if Path(splits_path).exists():
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        
        # Process each split
        for split_name in ['train', 'val', 'test']:
            print(f"\nProcessing {split_name} set...")
            file_paths = splits[split_name]['file_paths']
            labels = splits[split_name]['labels']
            
            output_path = f"data/processed/{split_name}_features.npz"
            X, y = extractor.process_dataset(file_paths, labels, output_path)
        
        # Save feature names
        feature_names = extractor.get_feature_names()
        print(f"\nTotal features: {len(feature_names)}")
    else:
        print(f"Please run data_loader.py first to create data splits")
