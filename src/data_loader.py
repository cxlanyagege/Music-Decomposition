"""
Data preprocessing module for audio files.
Handles loading, splitting, and organizing audio datasets.
"""

import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
import yaml
import json


class AudioDataLoader:
    """Load and preprocess audio files for instrument recognition."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the data loader.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sample_rate = self.config['audio']['sample_rate']
        self.duration = self.config['audio']['duration']
        self.offset = self.config['audio']['offset']
        
    def load_audio_file(self, file_path: str) -> np.ndarray:
        """
        Load an audio file and return the audio signal.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Audio signal as numpy array
        """
        try:
            # Load audio file with librosa
            audio, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                duration=self.duration,
                offset=self.offset
            )
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_dataset_from_directory(
        self, 
        data_dir: str,
        pattern: str = "**/*.wav"
    ) -> Tuple[List[str], List[str]]:
        """
        Load dataset from directory structure.
        Expects structure: data_dir/instrument_name/*.wav
        
        Args:
            data_dir: Root directory containing audio files
            pattern: Glob pattern for audio files
            
        Returns:
            Tuple of (file_paths, labels)
        """
        data_path = Path(data_dir)
        file_paths = []
        labels = []
        
        # Find all audio files
        audio_files = list(data_path.glob(pattern))
        
        if not audio_files:
            print(f"Warning: No audio files found in {data_dir}")
            return [], []
        
        for file_path in audio_files:
            # Get instrument label from parent directory name
            label = file_path.parent.name
            file_paths.append(str(file_path))
            labels.append(label)
        
        print(f"Found {len(file_paths)} audio files from {len(set(labels))} instruments")
        return file_paths, labels
    
    def create_train_val_test_split(
        self,
        file_paths: List[str],
        labels: List[str]
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            file_paths: List of file paths
            labels: List of corresponding labels
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing (paths, labels) tuples
        """
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        test_split = self.config['data']['test_split']
        random_seed = self.config['data']['random_seed']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            file_paths,
            labels,
            test_size=test_split,
            random_state=random_seed,
            stratify=labels
        )
        
        # Second split: separate train and validation from remaining data
        val_ratio = val_split / (train_split + val_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=y_temp
        )
        
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return splits
    
    def save_split_info(self, splits: Dict, output_path: str):
        """
        Save data split information to a JSON file.
        
        Args:
            splits: Dictionary containing split information
            output_path: Path to save the split info
        """
        split_info = {
            split_name: {
                'file_paths': paths,
                'labels': labels,
                'count': len(paths)
            }
            for split_name, (paths, labels) in splits.items()
        }
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"Split information saved to {output_path}")
    
    def get_label_mapping(self, labels: List[str]) -> Dict[str, int]:
        """
        Create a mapping from label names to integers.
        
        Args:
            labels: List of label names
            
        Returns:
            Dictionary mapping label names to integers
        """
        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        return label_to_idx


if __name__ == "__main__":
    # Example usage
    loader = AudioDataLoader()
    
    # Load dataset from directory
    data_dir = "data/raw"
    if os.path.exists(data_dir):
        file_paths, labels = loader.load_dataset_from_directory(data_dir)
        
        if file_paths:
            # Create splits
            splits = loader.create_train_val_test_split(file_paths, labels)
            
            # Save split information
            loader.save_split_info(splits, "data/processed/splits.json")
            
            # Create and save label mapping
            label_mapping = loader.get_label_mapping(labels)
            print(f"\nLabel mapping: {label_mapping}")
    else:
        print(f"Please create directory '{data_dir}' and add your audio files")
        print("Expected structure: data/raw/instrument_name/*.wav")
