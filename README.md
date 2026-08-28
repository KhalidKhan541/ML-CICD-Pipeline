# ML CI/CD Pipeline

A complete CI/CD pipeline for ML models using GitHub Actions with automated retraining, validation gates, and controlled promotion to production.

## Overview

This pipeline automates the entire ML lifecycle:

1. **Trigger**: Detects data changes or manual dispatch
2. **Train**: Retrains model on new data
3. **Validate**: Runs comprehensive validation gates
4. **Promote**: Deploys to production only if all gates pass

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GitHub Actions Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Trigger    │───▶│    Train     │───▶│   Validate   │      │
│  │              │    │              │    │              │      │
│  │ • Data push  │    │ • Load data  │    │ • Accuracy   │      │
│  │ • Manual     │    │ • Train model│    │ • Fairness   │      │
│  │ • Schedule   │    │ • Save model │    │ • Quality    │      │
│  └──────────────┘    └──────────────┘    │ • Performance│      │
│                                          └──────┬───────┘      │
│                                                 │              │
│                                    ┌────────────┴────────────┐ │
│                                    │                         │ │
│                                    ▼                         ▼ │
│                           ┌──────────────┐          ┌─────────┐ │
│                           │   Passed     │          │ Failed  │ │
│                           │              │          │         │ │
│                           │ • Backup     │          │ • Alert │ │
│                           │ • Canary     │          │ • Block │ │
│                           │ • Promote    │          │         │ │
│                           └──────────────┘          └─────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Automated Retraining**: Triggered by data changes
- **Validation Gates**: Accuracy, fairness, data quality, performance
- **Canary Deployment**: Safe rollout with monitoring
- **Automatic Rollback**: On failure detection
- **Model Registry**: Version control for models
- **Experiment Tracking**: MLflow integration
- **Data Versioning**: DVC support

## Quick Start

### Prerequisites

- GitHub repository
- Python 3.9+
- DVC (optional, for data versioning)
- MLflow server (optional, for tracking)

### Installation

```bash
# Clone the repository
git clone https://github.com/KhalidKhan541/ML-CICD-Pipeline.git
cd ML-CICD-Pipeline

# Install dependencies
pip install -r requirements.txt

# Initialize DVC (optional)
dvc init
dvc remote add -d origin https://github.com/KhalidKhan541/ML-CICD-Pipeline.git
```

### Configuration

1. **Set GitHub Secrets**:
   - `MLFLOW_TRACKING_URI`: MLflow server URL
   - `DVC_REMOTE_URL`: DVC remote URL
   - `AWS_ACCESS_KEY_ID`: AWS credentials (if using S3)
   - `AWS_SECRET_ACCESS_KEY`: AWS secret key

2. **Configure Validation Thresholds**:
   Edit `configs/validation_config.yaml`:

   ```yaml
   accuracy:
     min_accuracy: 0.75
     min_precision: 0.70
   
   fairness:
     protected_attributes:
       - "gender"
     max_disparity: 0.20
   ```

3. **Configure Promotion Rules**:
   Edit `configs/promotion_config.yaml`:

   ```yaml
   canary:
     default_traffic_percent: 10
     default_duration_minutes: 30
   
   thresholds:
     max_error_rate: 0.05
     max_latency_ms: 200
   ```

### Usage

#### Automatic Trigger

The pipeline runs automatically when:
- Data files in `data/` are modified and pushed
- Manual dispatch via GitHub UI

#### Manual Trigger

```bash
# Via GitHub CLI
gh workflow run train.yml -f force_retrain=true -f model_type=classifier

# Via GitHub UI
# Actions > Train Model > Run workflow
```

#### Check Pipeline Status

```bash
gh run list --workflow=train.yml
gh run view <run-id>
```

## Validation Gates

### 1. Accuracy Gate

Ensures model meets minimum performance metrics:

| Metric | Default Threshold |
|--------|-------------------|
| Accuracy | 0.75 |
| Precision | 0.70 |
| Recall | 0.70 |
| F1 Score | 0.70 |
| ROC AUC | 0.75 |

### 2. Fairness Gate

Checks for bias across protected attributes:

| Check | Default Threshold |
|-------|-------------------|
| Max Disparity | 0.20 |
| Min Group Size | 50 |

Protected attributes:
- Gender
- Ethnicity
- Age group

### 3. Data Quality Gate

Validates input data quality:

| Check | Default Threshold |
|-------|-------------------|
| Max Null Ratio | 0.30 |
| Min Samples | 100 |
| Max Cardinality | 0.90 |

### 4. Performance Gate

Ensures model meets latency/throughput requirements:

| Metric | Default Threshold |
|--------|-------------------|
| Max Latency | 100ms |
| Min Throughput | 100 req/s |
| Max Model Size | 500MB |

## Promotion Process

### Step 1: Production Readiness Check

```python
- Model file exists
- Version not already in production
- No critical issues
```

### Step 2: Backup Current Production

```python
- Save current model to backup/
- Create backup metadata
```

### Step 3: Canary Deployment

```python
- Deploy to 10% of traffic
- Monitor for 30 minutes
- Check error rate, latency, throughput
```

### Step 4: Full Promotion

```python
- If canary passes, deploy to 100%
- Update production registry
- Notify stakeholders
```

### Step 5: Rollback (if needed)

```python
- Automatic rollback on canary failure
- Restore previous production model
- Send alerts
```

## Project Structure

```
ml-cicd-pipeline/
├── .github/
│   └── workflows/
│       ├── train.yml           # Training pipeline
│       ├── validate.yml        # Validation pipeline
│       └── promote.yml         # Promotion pipeline
├── src/
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py           # Model training
│   ├── validation/
│   │   ├── __init__.py
│   │   └── validation_gates.py # Validation gates
│   └── promotion/
│       ├── __init__.py
│       └── promoter.py        # Model promotion
├── configs/
│   ├── training_config.yaml
│   ├── validation_config.yaml
│   └── promotion_config.yaml
├── tests/
│   └── test_validation.py
├── data/                      # Training data
├── models/                    # Trained models
├── .dvc/
│   └── config                 # DVC configuration
├── requirements.txt
└── README.md
```

## GitHub Actions Workflows

### Train Model (`train.yml`)

**Triggers:**
- Push to `main` with changes in `data/`
- Manual dispatch

**Steps:**
1. Check for data changes
2. Pull latest data (DVC)
3. Train model
4. Save model artifacts
5. Log to MLflow

### Validate Model (`validate.yml`)

**Triggers:**
- After training completes
- Manual dispatch

**Steps:**
1. Run accuracy gate
2. Run fairness gate
3. Run data quality gate
4. Run performance gate
5. Aggregate results
6. Trigger promotion if passed

### Promote to Production (`promote.yml`)

**Triggers:**
- After validation passes
- Manual dispatch

**Steps:**
1. Check production readiness
2. Backup current production
3. Deploy canary (10% traffic)
4. Monitor canary (30 min)
5. Promote to full production
6. Update registry
7. Notify stakeholders

## Configuration Reference

### Training Config

```yaml
model_type: "random_forest"
model_params:
  random_forest:
    n_estimators: 100
    max_depth: 10
```

### Validation Config

```yaml
accuracy:
  min_accuracy: 0.75

fairness:
  protected_attributes: ["gender"]
  max_disparity: 0.20

performance:
  max_latency_ms: 100
```

### Promotion Config

```yaml
canary:
  default_traffic_percent: 10
  default_duration_minutes: 30

rollback:
  auto_rollback_on_failure: true
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_validation.py::TestAccuracyGate

# Run with coverage
pytest tests/ --cov=src
```

## Monitoring

### MLflow Integration

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
with mlflow.start_run():
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_metric("accuracy", 0.95)
```

### DVC Metrics

```bash
# Show metrics
dvc metrics show

# Compare experiments
dvc experiments compare
```

## Troubleshooting

### Common Issues

1. **Training fails**: Check data format and dependencies
2. **Validation fails**: Review thresholds in config
3. **Canary fails**: Check service health and metrics
4. **DVC issues**: Verify remote credentials

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m src.training.train --config configs/training_config.yaml -v
```

## License

MIT License