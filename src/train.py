"""
Model training module for instrument recognition.
Trains SVM, Random Forest, and XGBoost models.
"""

import numpy as np
import yaml
import json
import joblib
from pathlib import Path
from typing import Dict, Tuple
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class InstrumentClassifier:
    """Train and evaluate models for instrument recognition."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the classifier.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
        
    def load_features(self, feature_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features from .npz file.
        
        Args:
            feature_path: Path to features file
            
        Returns:
            Tuple of (features, labels)
        """
        data = np.load(feature_path)
        X = data['features']
        y = data['labels']
        return X, y
    
    def prepare_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Tuple:
        """
        Prepare data for training (scaling and encoding).
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Tuple of prepared data
        """
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Fit label encoder on training labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_val_encoded = self.label_encoder.transform(y_val)
            return X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded
        
        return X_train_scaled, y_train_encoded
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray) -> SVC:
        """
        Train SVM classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained SVM model
        """
        print("\nTraining SVM...")
        svm_config = self.config['training']['svm']
        
        model = SVC(
            kernel=svm_config['kernel'],
            C=svm_config['C'],
            gamma=svm_config['gamma'],
            probability=True,
            verbose=True
        )
        
        model.fit(X_train, y_train)
        self.models['svm'] = model
        
        print("SVM training completed")
        return model
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> RandomForestClassifier:
        """
        Train Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained Random Forest model
        """
        print("\nTraining Random Forest...")
        rf_config = self.config['training']['random_forest']
        
        model = RandomForestClassifier(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            min_samples_split=rf_config['min_samples_split'],
            random_state=rf_config['random_state'],
            verbose=1,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        print("Random Forest training completed")
        return model
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained XGBoost model
        """
        print("\nTraining XGBoost...")
        xgb_config = self.config['training']['xgboost']
        
        model = xgb.XGBClassifier(
            n_estimators=xgb_config['n_estimators'],
            max_depth=xgb_config['max_depth'],
            learning_rate=xgb_config['learning_rate'],
            random_state=xgb_config['random_state'],
            verbosity=1,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        print("XGBoost training completed")
        return model
    
    def evaluate_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        split_name: str = "test"
    ) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            model_name: Name of the model
            split_name: Name of the split (train/val/test)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Classification report
        report = classification_report(
            y,
            y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        results = {
            'model': model_name,
            'split': split_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        print(f"\n{model_name} - {split_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return results
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        model_name: str,
        output_dir: str
    ):
        """
        Plot and save confusion matrix.
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            output_dir: Directory to save plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'{model_name}_confusion_matrix.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {output_path}")
    
    def save_model(self, model_name: str, output_dir: str):
        """
        Save trained model and preprocessing objects.
        
        Args:
            model_name: Name of the model to save
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / f'{model_name}_model.pkl'
        joblib.dump(self.models[model_name], model_path)
        
        # Save scaler and label encoder
        scaler_path = output_path / 'scaler.pkl'
        encoder_path = output_path / 'label_encoder.pkl'
        
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"Model saved to {model_path}")
    
    def save_results(self, output_path: str):
        """
        Save evaluation results to JSON.
        
        Args:
            output_path: Path to save results
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ):
        """
        Train all models specified in config.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
        """
        # Prepare data
        X_train_prep, y_train_prep, X_val_prep, y_val_prep = self.prepare_data(
            X_train, y_train, X_val, y_val
        )
        X_test_prep = self.scaler.transform(X_test)
        y_test_prep = self.label_encoder.transform(y_test)
        
        model_types = self.config['training']['models']
        
        # Train each model
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()}")
            print(f"{'='*60}")
            
            if model_type == 'svm':
                model = self.train_svm(X_train_prep, y_train_prep)
            elif model_type == 'random_forest':
                model = self.train_random_forest(X_train_prep, y_train_prep)
            elif model_type == 'xgboost':
                model = self.train_xgboost(X_train_prep, y_train_prep)
            else:
                print(f"Unknown model type: {model_type}")
                continue
            
            # Evaluate on all splits
            train_results = self.evaluate_model(
                model, X_train_prep, y_train_prep, model_type, 'train'
            )
            val_results = self.evaluate_model(
                model, X_val_prep, y_val_prep, model_type, 'val'
            )
            test_results = self.evaluate_model(
                model, X_test_prep, y_test_prep, model_type, 'test'
            )
            
            # Store results
            self.results[model_type] = {
                'train': train_results,
                'val': val_results,
                'test': test_results
            }
            
            # Plot confusion matrix
            results_dir = self.config['output']['results_dir']
            cm = np.array(test_results['confusion_matrix'])
            self.plot_confusion_matrix(cm, model_type, results_dir)
            
            # Save model
            model_dir = self.config['output']['model_dir']
            self.save_model(model_type, model_dir)


if __name__ == "__main__":
    # Initialize classifier
    classifier = InstrumentClassifier()
    
    # Load features
    print("Loading features...")
    X_train, y_train = classifier.load_features("data/processed/train_features.npz")
    X_val, y_val = classifier.load_features("data/processed/val_features.npz")
    X_test, y_test = classifier.load_features("data/processed/test_features.npz")
    
    print(f"\nDataset sizes:")
    print(f"Train: {X_train.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    # Train all models
    classifier.train_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Save results
    results_path = "results/evaluation_results.json"
    classifier.save_results(results_path)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
