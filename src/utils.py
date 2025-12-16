"""
Utility functions for the music instrument recognition project.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Dict, output_path: str):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        output_path: Output file path
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(file_path: str) -> Dict:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def create_directory_structure():
    """Create necessary directories for the project."""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'results',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def get_audio_files(directory: str, extensions: list = None) -> list:
    """
    Get list of audio files in directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions (default: wav, mp3, flac)
        
    Returns:
        List of audio file paths
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.ogg']
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(Path(directory).rglob(f'*{ext}'))
    
    return [str(f) for f in audio_files]


if __name__ == "__main__":
    # Create directory structure
    create_directory_structure()
    print("\nProject directory structure created!")
