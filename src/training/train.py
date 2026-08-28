import os
import json
import yaml
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config.get('data_path', 'data/'))
    
    def load_data(self, split: str = 'train') -> pd.DataFrame:
        data_file = self.data_path / f'{split}.csv'
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {split} data: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if target_column is None:
            target_column = self.config.get('target_column', 'target')
        
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df
            y = None
        
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        
        X = X.fillna(X.median())
        
        return X, y

class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def get_model(self, model_type: str = None):
        if model_type is None:
            model_type = self.config.get('model_type', 'classifier')
        
        model_params = self.config.get('model_params', {})
        
        models = {
            'random_forest': RandomForestClassifier(**model_params.get('random_forest', {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            })),
            'gradient_boosting': GradientBoostingClassifier(**model_params.get('gradient_boosting', {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42
            })),
            'logistic_regression': LogisticRegression(**model_params.get('logistic_regression', {
                'max_iter': 1000,
                'random_state': 42
            }))
        }
        
        return models.get(model_type, models['random_forest'])
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              model_type: str = None) -> Dict[str, Any]:
        logger.info(f"Training {model_type or self.config.get('model_type', 'classifier')} model...")
        
        self.feature_names = X_train.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = self.get_model(model_type)
        self.model.fit(X_scaled, y_train)
        
        train_score = self.model.score(X_scaled, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        return {
            'training_accuracy': train_score,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X_test)
        
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        if y_proba is not None and len(np.unique(y_test)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        return metrics
    
    def save(self, output_path: Path, metrics: Dict[str, Any]):
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = output_path / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'config': self.config
            }, f)
        
        metrics_path = output_path / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        scaler_path = output_path / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Model saved to {output_path}")
    
    @classmethod
    def load(cls, model_path: Path) -> 'ModelTrainer':
        with open(model_path / 'model.pkl', 'rb') as f:
            data = pickle.load(f)
        
        trainer = cls(data['config'])
        trainer.model = data['model']
        trainer.scaler = data['scaler']
        trainer.feature_names = data['feature_names']
        
        return trainer

def run_training(config_path: str, model_type: str = None, output_dir: str = None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_type:
        config['model_type'] = model_type
    
    data_loader = DataLoader(config)
    
    train_df = data_loader.load_data('train')
    X_train, y_train = data_loader.prepare_features(train_df)
    
    test_df = data_loader.load_data('test')
    X_test, y_test = data_loader.prepare_features(test_df)
    
    trainer = ModelTrainer(config)
    train_metrics = trainer.train(X_train, y_train, model_type)
    
    test_metrics = trainer.evaluate(X_test, y_test)
    
    all_metrics = {**train_metrics, **test_metrics}
    
    output_path = Path(output_dir or config.get('output_dir', 'models/latest'))
    trainer.save(output_path, all_metrics)
    
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        with mlflow.start_run(run_name=f'train-{datetime.now().strftime("%Y%m%d-%H%M%S")}'):
            mlflow.log_params({
                'model_type': model_type or config.get('model_type', 'classifier'),
                'n_features': X_train.shape[1],
                'n_train_samples': X_train.shape[0],
                'n_test_samples': X_test.shape[0]
            })
            mlflow.log_metrics(all_metrics)
            mlflow.sklearn.log_model(trainer.model, 'model')
    
    logger.info(f"Training complete. Metrics: {all_metrics}")
    return all_metrics

def main():
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--config', required=True, help='Training config path')
    parser.add_argument('--model-type', default=None, help='Model type to train')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    metrics = run_training(args.config, args.model_type, args.output_dir)
    print(json.dumps(metrics, indent=2, default=str))

if __name__ == '__main__':
    main()