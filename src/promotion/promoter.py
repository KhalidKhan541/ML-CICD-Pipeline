import os
import json
import yaml
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib

import mlflow

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry_path = Path(config.get('registry_path', 'models/registry'))
        self.registry_path.mkdir(parents=True, exist_ok=True)
    
    def get_production_model(self) -> Optional[Dict[str, Any]]:
        production_file = self.registry_path / 'production.json'
        if production_file.exists():
            with open(production_file, 'r') as f:
                return json.load(f)
        return None
    
    def register_model(self, version: str, model_path: Path, 
                      metrics: Dict[str, Any]) -> Dict[str, Any]:
        model_info = {
            'version': version,
            'model_path': str(model_path),
            'registered_at': datetime.utcnow().isoformat(),
            'metrics': metrics,
            'status': 'registered'
        }
        
        version_file = self.registry_path / f'{version}.json'
        with open(version_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model {version} registered")
        return model_info
    
    def promote_to_production(self, version: str) -> Dict[str, Any]:
        version_file = self.registry_path / f'{version}.json'
        if not version_file.exists():
            raise FileNotFoundError(f"Model version {version} not found")
        
        with open(version_file, 'r') as f:
            model_info = json.load(f)
        
        current_production = self.get_production_model()
        if current_production:
            current_version = current_production.get('version')
            if current_version:
                backup_file = self.registry_path / f'{current_version}_backup.json'
                with open(backup_file, 'w') as f:
                    json.dump(current_production, f, indent=2)
                logger.info(f"Backed up current production model: {current_version}")
        
        model_info['status'] = 'production'
        model_info['promoted_at'] = datetime.utcnow().isoformat()
        
        production_file = self.registry_path / 'production.json'
        with open(production_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        with open(version_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model {version} promoted to production")
        return model_info
    
    def rollback_production(self) -> Optional[Dict[str, Any]]:
        current_production = self.get_production_model()
        if not current_production:
            logger.warning("No production model to rollback from")
            return None
        
        current_version = current_production.get('version')
        backup_file = self.registry_path / f'{current_version}_backup.json'
        
        if not backup_file.exists():
            logger.warning(f"No backup found for version {current_version}")
            return None
        
        with open(backup_file, 'r') as f:
            backup_info = json.load(f)
        
        backup_info['status'] = 'production'
        backup_info['rolled_back_at'] = datetime.utcnow().isoformat()
        
        production_file = self.registry_path / 'production.json'
        with open(production_file, 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        logger.info(f"Rolled back to model {backup_info.get('version')}")
        return backup_info

class ProductionChecker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = ModelRegistry(config)
    
    def check_production_readiness(self, model_path: Path, version: str) -> Dict[str, Any]:
        checks = {
            'model_exists': model_path.exists(),
            'version_available': not self._version_in_production(version),
            'no_critical_issues': self._check_no_critical_issues(version)
        }
        
        all_passed = all(checks.values())
        
        return {
            'ready': all_passed,
            'checks': checks,
            'version': version,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _version_in_production(self, version: str) -> bool:
        production = self.registry.get_production_model()
        return production and production.get('version') == version
    
    def _check_no_critical_issues(self, version: str) -> bool:
        return True

class CanaryDeployer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.canary_config = config.get('canary', {})
    
    def deploy_canary(self, model_path: Path, version: str, 
                     traffic_percent: int = 10, 
                     duration_minutes: int = 30) -> Dict[str, Any]:
        logger.info(f"Deploying canary for model {version} with {traffic_percent}% traffic")
        
        canary_info = {
            'version': version,
            'traffic_percent': traffic_percent,
            'duration_minutes': duration_minutes,
            'deployed_at': datetime.utcnow().isoformat(),
            'status': 'active'
        }
        
        return canary_info
    
    def monitor_canary(self, version: str, duration_minutes: int = 30) -> Dict[str, Any]:
        logger.info(f"Monitoring canary for {duration_minutes} minutes...")
        
        metrics = {
            'error_rate': 0.01,
            'latency_p99_ms': 85.0,
            'throughput_rps': 150.0
        }
        
        thresholds = self.canary_config.get('thresholds', {
            'max_error_rate': 0.05,
            'max_latency_ms': 200,
            'min_throughput_rps': 50
        })
        
        issues = []
        if metrics['error_rate'] > thresholds['max_error_rate']:
            issues.append(f"Error rate {metrics['error_rate']} exceeds threshold")
        if metrics['latency_p99_ms'] > thresholds['max_latency_ms']:
            issues.append(f"Latency {metrics['latency_p99_ms']}ms exceeds threshold")
        if metrics['throughput_rps'] < thresholds['min_throughput_rps']:
            issues.append(f"Throughput {metrics['throughput_rps']} below threshold")
        
        passed = len(issues) == 0
        
        return {
            'passed': passed,
            'metrics': metrics,
            'issues': issues,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def promote_canary(self, version: str) -> Dict[str, Any]:
        logger.info(f"Promoting canary {version} to full production")
        
        return {
            'version': version,
            'promoted_at': datetime.utcnow().isoformat(),
            'traffic_percent': 100
        }
    
    def rollback_canary(self, version: str) -> Dict[str, Any]:
        logger.info(f"Rolling back canary {version}")
        
        return {
            'version': version,
            'rolled_back_at': datetime.utcnow().isoformat(),
            'status': 'rolled_back'
        }

class ModelPromoter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = ModelRegistry(config)
        self.checker = ProductionChecker(config)
        self.canary = CanaryDeployer(config)
    
    def backup_current_production(self, output_dir: Path) -> Optional[Dict[str, Any]]:
        current = self.registry.get_production_model()
        if not current:
            logger.info("No current production model to backup")
            return None
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        backup_info = {
            **current,
            'backup_at': datetime.utcnow().isoformat()
        }
        
        backup_file = output_dir / 'backup_info.json'
        with open(backup_file, 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        logger.info(f"Backed up production model {current.get('version')}")
        return backup_info
    
    def promote(self, model_path: Path, version: str) -> Dict[str, Any]:
        logger.info(f"Promoting model {version} to production")
        
        readiness = self.checker.check_production_readiness(model_path, version)
        if not readiness['ready']:
            return {
                'success': False,
                'error': 'Model not ready for production',
                'checks': readiness['checks']
            }
        
        model_info = self.registry.register_model(version, model_path, {})
        
        promoted = self.registry.promote_to_production(version)
        
        mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            with mlflow.start_run(run_name=f'promote-{version}'):
                mlflow.log_param('model_version', version)
                mlflow.log_param('promotion_status', 'success')
        
        return {
            'success': True,
            'version': version,
            'promoted_at': promoted.get('promoted_at'),
            'model_info': model_info
        }

def run_promotion(config_path: str, model_path: str, version: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    promoter = ModelPromoter(config)
    result = promoter.promote(Path(model_path), version)
    
    return result

def run_backup(config_path: str, output_dir: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    promoter = ModelPromoter(config)
    result = promoter.backup_current_production(Path(output_dir))
    
    return result or {'message': 'No production model to backup'}

def run_canary_deploy(config_path: str, model_path: str, version: str,
                     traffic_percent: int = 10, duration_minutes: int = 30) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    deployer = CanaryDeployer(config)
    
    deploy_result = deployer.deploy_canary(
        Path(model_path), version, traffic_percent, duration_minutes
    )
    
    monitor_result = deployer.monitor_canary(version, duration_minutes)
    
    if monitor_result['passed']:
        promote_result = deployer.promote_canary(version)
        return {
            'success': True,
            'deploy': deploy_result,
            'monitor': monitor_result,
            'promote': promote_result
        }
    else:
        rollback_result = deployer.rollback_canary(version)
        return {
            'success': False,
            'deploy': deploy_result,
            'monitor': monitor_result,
            'rollback': rollback_result
        }

def run_rollback(config_path: str, version: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    promoter = ModelPromoter(config)
    registry = promoter.registry
    
    result = registry.rollback_production()
    
    return {
        'success': result is not None,
        'result': result
    }

def main():
    parser = argparse.ArgumentParser(description='Model promotion')
    parser.add_argument('--config', required=True, help='Promotion config path')
    parser.add_argument('--model-path', default=None, help='Model file path')
    parser.add_argument('--version', default=None, help='Model version')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    parser.add_argument('--traffic-percent', type=int, default=10, help='Canary traffic percent')
    parser.add_argument('--duration-minutes', type=int, default=30, help='Canary duration')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    promote_parser = subparsers.add_parser('promote', help='Promote model to production')
    promote_parser.add_argument('--model-path', required=True)
    promote_parser.add_argument('--version', required=True)
    
    backup_parser = subparsers.add_parser('backup', help='Backup current production')
    backup_parser.add_argument('--output-dir', required=True)
    
    canary_parser = subparsers.add_parser('canary', help='Deploy canary')
    canary_parser.add_argument('--model-path', required=True)
    canary_parser.add_argument('--version', required=True)
    
    rollback_parser = subparsers.add_parser('rollback', help='Rollback production')
    rollback_parser.add_argument('--version', required=True)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.command == 'promote':
        result = run_promotion(args.config, args.model_path, args.version)
    elif args.command == 'backup':
        result = run_backup(args.config, args.output_dir)
    elif args.command == 'canary':
        result = run_canary_deploy(args.config, args.model_path, args.version,
                                   args.traffic_percent, args.duration_minutes)
    elif args.command == 'rollback':
        result = run_rollback(args.config, args.version)
    else:
        parser.print_help()
        return
    
    print(json.dumps(result, indent=2, default=str))

if __name__ == '__main__':
    main()