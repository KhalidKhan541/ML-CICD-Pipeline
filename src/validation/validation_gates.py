import json
import yaml
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    gate_name: str
    passed: bool
    metrics: Dict[str, Any]
    threshold_violations: List[Dict[str, Any]]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gate_name': self.gate_name,
            'passed': self.passed,
            'metrics': self.metrics,
            'threshold_violations': self.threshold_violations,
            'timestamp': self.timestamp
        }

class AccuracyGate:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config.get('accuracy', {})
    
    def validate(self, model_path: Path, data_path: Path) -> ValidationResult:
        logger.info("Running accuracy validation gate...")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        
        test_data = pd.read_csv(data_path / 'test.csv')
        
        target_column = self.config.get('target_column', 'target')
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        
        categorical_columns = X_test.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)
        
        X_test = X_test.fillna(X_test.median())
        
        X_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_scaled)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        }
        
        violations = []
        for metric_name, threshold in self.thresholds.items():
            if metric_name in metrics:
                if metrics[metric_name] < threshold:
                    violations.append({
                        'metric': metric_name,
                        'value': metrics[metric_name],
                        'threshold': threshold,
                        'difference': threshold - metrics[metric_name]
                    })
        
        passed = len(violations) == 0
        
        return ValidationResult(
            gate_name='accuracy',
            passed=passed,
            metrics=metrics,
            threshold_violations=violations,
            timestamp=datetime.utcnow().isoformat()
        )

class FairnessGate:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fairness_config = config.get('fairness', {})
        self.protected_attributes = self.fairness_config.get('protected_attributes', [])
        self.max_disparity = self.fairness_config.get('max_disparity', 0.2)
    
    def validate(self, model_path: Path, data_path: Path) -> ValidationResult:
        logger.info("Running fairness validation gate...")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        
        test_data = pd.read_csv(data_path / 'test.csv')
        
        target_column = self.config.get('target_column', 'target')
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        
        categorical_columns = X_test.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)
        
        X_test = X_test.fillna(X_test.median())
        
        X_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_scaled)
        
        metrics = {}
        violations = []
        
        for attr in self.protected_attributes:
            if attr in test_data.columns:
                groups = test_data[attr].unique()
                group_metrics = {}
                
                for group in groups:
                    mask = test_data[attr] == group
                    if mask.sum() > 0:
                        group_acc = accuracy_score(y_test[mask], y_pred[mask])
                        group_metrics[str(group)] = float(group_acc)
                
                if len(group_metrics) > 1:
                    values = list(group_metrics.values())
                    disparity = max(values) - min(values)
                    
                    metrics[f'{attr}_disparity'] = float(disparity)
                    metrics[f'{attr}_group_metrics'] = group_metrics
                    
                    if disparity > self.max_disparity:
                        violations.append({
                            'attribute': attr,
                            'disparity': float(disparity),
                            'max_allowed': self.max_disparity,
                            'group_metrics': group_metrics
                        })
        
        overall_accuracy = accuracy_score(y_test, y_pred)
        metrics['overall_accuracy'] = float(overall_accuracy)
        
        passed = len(violations) == 0
        
        return ValidationResult(
            gate_name='fairness',
            passed=passed,
            metrics=metrics,
            threshold_violations=violations,
            timestamp=datetime.utcnow().isoformat()
        )

class DataQualityGate:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_config = config.get('data_quality', {})
        self.max_null_ratio = self.quality_config.get('max_null_ratio', 0.3)
        self.max_cardinality_ratio = self.quality_config.get('max_cardinality_ratio', 0.9)
        self.min_samples = self.quality_config.get('min_samples', 100)
    
    def validate(self, data_path: Path) -> ValidationResult:
        logger.info("Running data quality validation gate...")
        
        test_data = pd.read_csv(data_path / 'test.csv')
        
        metrics = {
            'n_samples': len(test_data),
            'n_features': len(test_data.columns),
            'null_ratio': float(test_data.isnull().sum().sum() / test_data.size),
        }
        
        violations = []
        
        if len(test_data) < self.min_samples:
            violations.append({
                'check': 'min_samples',
                'value': len(test_data),
                'threshold': self.min_samples
            })
        
        null_ratios = test_data.isnull().mean()
        high_null_columns = null_ratios[null_ratios > self.max_null_ratio].index.tolist()
        if high_null_columns:
            violations.append({
                'check': 'null_ratio',
                'columns': high_null_columns,
                'max_ratio': float(null_ratios[high_null_columns].max()),
                'threshold': self.max_null_ratio
            })
        
        for col in test_data.select_dtypes(include=['object', 'category']).columns:
            cardinality_ratio = test_data[col].nunique() / len(test_data)
            if cardinality_ratio > self.max_cardinality_ratio:
                violations.append({
                    'check': 'cardinality',
                    'column': col,
                    'ratio': float(cardinality_ratio),
                    'threshold': self.max_cardinality_ratio
                })
        
        passed = len(violations) == 0
        
        return ValidationResult(
            gate_name='data_quality',
            passed=passed,
            metrics=metrics,
            threshold_violations=violations,
            timestamp=datetime.utcnow().isoformat()
        )

class PerformanceGate:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.perf_config = config.get('performance', {})
        self.max_latency_ms = self.perf_config.get('max_latency_ms', 100)
        self.min_throughput_rps = self.perf_config.get('min_throughput_rps', 100)
    
    def validate(self, model_path: Path) -> ValidationResult:
        logger.info("Running performance validation gate...")
        
        import time
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        
        n_features = len(model_data.get('feature_names', []))
        if n_features == 0:
            n_features = 10
        
        test_sample = np.random.randn(1, n_features)
        
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            X_scaled = scaler.transform(test_sample)
            model.predict(X_scaled)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        throughput_rps = 1000 / avg_latency if avg_latency > 0 else 0
        
        metrics = {
            'avg_latency_ms': float(avg_latency),
            'p95_latency_ms': float(p95_latency),
            'throughput_rps': float(throughput_rps)
        }
        
        violations = []
        
        if avg_latency > self.max_latency_ms:
            violations.append({
                'check': 'latency',
                'value': float(avg_latency),
                'threshold': self.max_latency_ms
            })
        
        if throughput_rps < self.min_throughput_rps:
            violations.append({
                'check': 'throughput',
                'value': float(throughput_rps),
                'threshold': self.min_throughput_rps
            })
        
        passed = len(violations) == 0
        
        return ValidationResult(
            gate_name='performance',
            passed=passed,
            metrics=metrics,
            threshold_violations=violations,
            timestamp=datetime.utcnow().isoformat()
        )

def run_validation(config_path: str, model_path: str, data_path: str = None, 
                   output_path: str = None) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = Path(model_path)
    data_path = Path(data_path) if data_path else Path(config.get('data_path', 'data/'))
    
    results = {}
    
    accuracy_gate = AccuracyGate(config)
    results['accuracy'] = accuracy_gate.validate(model_path, data_path).to_dict()
    
    fairness_gate = FairnessGate(config)
    results['fairness'] = fairness_gate.validate(model_path, data_path).to_dict()
    
    data_quality_gate = DataQualityGate(config)
    results['data_quality'] = data_quality_gate.validate(data_path).to_dict()
    
    performance_gate = PerformanceGate(config)
    results['performance'] = performance_gate.validate(model_path).to_dict()
    
    all_passed = all(r['passed'] for r in results.values())
    results['summary'] = {
        'all_passed': all_passed,
        'gates_passed': sum(1 for r in results.values() if r['passed']),
        'gates_total': len(results) - 1,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run validation gates')
    parser.add_argument('--config', required=True, help='Validation config path')
    parser.add_argument('--model-path', required=True, help='Model file path')
    parser.add_argument('--data-path', default=None, help='Data directory path')
    parser.add_argument('--output', default=None, help='Output JSON path')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    results = run_validation(args.config, args.model_path, args.data_path, args.output)
    
    print(json.dumps(results['summary'], indent=2))
    
    if not results['summary']['all_passed']:
        exit(1)

if __name__ == '__main__':
    main()