import pytest
import json
import yaml
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.validation.validation_gates import (
    AccuracyGate,
    FairnessGate,
    DataQualityGate,
    PerformanceGate,
    ValidationResult,
    run_validation
)
from src.promotion.promoter import (
    ModelRegistry,
    ProductionChecker,
    CanaryDeployer,
    ModelPromoter
)

@pytest.fixture
def sample_config():
    return {
        'target_column': 'target',
        'accuracy': {
            'min_accuracy': 0.75,
            'min_precision': 0.70,
            'min_recall': 0.70,
            'min_f1_score': 0.70
        },
        'fairness': {
            'protected_attributes': ['gender'],
            'max_disparity': 0.20
        },
        'data_quality': {
            'max_null_ratio': 0.30,
            'min_samples': 100
        },
        'performance': {
            'max_latency_ms': 100,
            'min_throughput_rps': 100
        }
    }

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 200
    
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.uniform(0, 10, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    })
    
    return data

@pytest.fixture
def sample_model(tmp_path, sample_data):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import pickle
    
    X = sample_data.drop(columns=['target', 'gender'])
    y = sample_data['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_scaled, y)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'config': {'target_column': 'target'}
    }
    
    model_path = tmp_path / 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_path

@pytest.fixture
def sample_test_data(tmp_path, sample_data):
    test_data = sample_data.sample(100, random_state=42)
    test_path = tmp_path / 'test.csv'
    test_data.to_csv(test_path, index=False)
    return tmp_path

class TestAccuracyGate:
    def test_validate_passes_with_good_model(self, sample_config, sample_model, sample_test_data):
        gate = AccuracyGate(sample_config)
        result = gate.validate(sample_model, sample_test_data)
        
        assert isinstance(result, ValidationResult)
        assert result.gate_name == 'accuracy'
        assert 'accuracy' in result.metrics
        assert 'precision' in result.metrics
        assert 'recall' in result.metrics
        assert 'f1_score' in result.metrics
    
    def test_validate_fails_with_low_accuracy(self, sample_config, sample_model, sample_test_data):
        sample_config['accuracy']['min_accuracy'] = 0.99
        
        gate = AccuracyGate(sample_config)
        result = gate.validate(sample_model, sample_test_data)
        
        assert not result.passed
        assert len(result.threshold_violations) > 0

class TestFairnessGate:
    def test_validate_checks_protected_attributes(self, sample_config, sample_model, sample_test_data):
        gate = FairnessGate(sample_config)
        result = gate.validate(sample_model, sample_test_data)
        
        assert isinstance(result, ValidationResult)
        assert result.gate_name == 'fairness'
        assert 'overall_accuracy' in result.metrics
    
    def test_validate_passes_with_fair_model(self, sample_config, sample_model, sample_test_data):
        gate = FairnessGate(sample_config)
        result = gate.validate(sample_model, sample_test_data)
        
        assert isinstance(result.passed, bool)

class TestDataQualityGate:
    def test_validate_checks_data_quality(self, sample_config, sample_test_data):
        gate = DataQualityGate(sample_config)
        result = gate.validate(sample_test_data)
        
        assert isinstance(result, ValidationResult)
        assert result.gate_name == 'data_quality'
        assert 'n_samples' in result.metrics
        assert 'n_features' in result.metrics
    
    def test_validate_fails_with_insufficient_samples(self, sample_config, tmp_path):
        small_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [0, 1, 0]
        })
        small_path = tmp_path / 'test.csv'
        small_data.to_csv(small_path, index=False)
        
        gate = DataQualityGate(sample_config)
        result = gate.validate(tmp_path)
        
        assert not result.passed

class TestPerformanceGate:
    def test_validate_checks_latency(self, sample_config, sample_model):
        gate = PerformanceGate(sample_config)
        result = gate.validate(sample_model)
        
        assert isinstance(result, ValidationResult)
        assert result.gate_name == 'performance'
        assert 'avg_latency_ms' in result.metrics
        assert 'throughput_rps' in result.metrics

class TestValidationResult:
    def test_to_dict(self):
        result = ValidationResult(
            gate_name='test',
            passed=True,
            metrics={'accuracy': 0.95},
            threshold_violations=[],
            timestamp=datetime.utcnow().isoformat()
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['gate_name'] == 'test'
        assert result_dict['passed'] is True
        assert result_dict['metrics']['accuracy'] == 0.95

class TestModelRegistry:
    def test_register_and_get_model(self, tmp_path):
        config = {'registry_path': str(tmp_path)}
        registry = ModelRegistry(config)
        
        model_info = registry.register_model(
            'v1.0',
            tmp_path / 'model.pkl',
            {'accuracy': 0.95}
        )
        
        assert model_info['version'] == 'v1.0'
        assert model_info['status'] == 'registered'
    
    def test_promote_to_production(self, tmp_path):
        config = {'registry_path': str(tmp_path)}
        registry = ModelRegistry(config)
        
        registry.register_model('v1.0', tmp_path / 'model.pkl', {'accuracy': 0.95})
        promoted = registry.promote_to_production('v1.0')
        
        assert promoted['status'] == 'production'
        
        production = registry.get_production_model()
        assert production['version'] == 'v1.0'

class TestCanaryDeployer:
    def test_deploy_canary(self, tmp_path):
        config = {'canary': {'thresholds': {'max_error_rate': 0.05}}}
        deployer = CanaryDeployer(config)
        
        result = deployer.deploy_canary(tmp_path / 'model.pkl', 'v1.0', 10, 30)
        
        assert result['version'] == 'v1.0'
        assert result['traffic_percent'] == 10
    
    def test_monitor_canary_passes(self):
        config = {'canary': {'thresholds': {'max_error_rate': 0.05}}}
        deployer = CanaryDeployer(config)
        
        result = deployer.monitor_canary('v1.0', 30)
        
        assert result['passed'] is True

def run_validation_integration(tmp_path, sample_config, sample_model, sample_test_data):
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    results = run_validation(
        str(config_path),
        str(sample_model),
        str(sample_test_data),
        str(tmp_path / 'results.json')
    )
    
    assert 'accuracy' in results
    assert 'fairness' in results
    assert 'data_quality' in results
    assert 'performance' in results
    assert 'summary' in results
    assert isinstance(results['summary']['all_passed'], bool)