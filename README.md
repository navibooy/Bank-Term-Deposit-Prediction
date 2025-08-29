# Bank Term Deposit Prediction - MLOps Pipeline

A comprehensive MLOps pipeline for predicting bank term deposit subscriptions using CatBoost, featuring automated training, drift detection, and deployment with Airflow orchestration.

## Project Overview

This project implements an end-to-end machine learning operations (MLOps) pipeline to predict customer responses to bank term deposit marketing campaigns. The system leverages modern MLOps tools and practices to ensure reliable, scalable, and maintainable ML workflows.

**Business Problem**: Predict which customers are most likely to subscribe to term deposit products, enabling banks to optimize marketing campaign effectiveness and resource allocation.

**Model**: CatBoost classifier with automated hyperparameter tuning and model validation.

## Technology Stack

- **ML Framework**: CatBoost, Scikit-learn
- **Orchestration**: Apache Airflow
- **Experiment Tracking**: MLflow
- **API**: FastAPI
- **Monitoring**: Evidently AI for drift detection
- **Containerization**: Docker Compose
- **Database**: PostgreSQL
- **Code Quality**: Pre-commit hooks, Ruff, isort

## Dataset Information

**Source**: [Kaggle Playground Series S5E8](https://www.kaggle.com/competitions/playground-series-s5e8)

**Problem Type**: Binary classification (term deposit subscription prediction)

**Dataset Size**:
- Training: 45,211 records
- Test: 30,141 records
- Features: 17 (16 input + 1 target)

**Target Distribution**: Imbalanced (88% no subscription, 12% subscription)

**Feature Categories**:
- **Demographics**: age, job, marital status, education
- **Financial**: balance, default status, housing/personal loans
- **Campaign**: contact method, duration, campaign frequency
- **Historical**: previous campaign outcomes, contact history

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Git

### 1. Clone Repository

```bash
git clone <repository-url>
cd Bank-Term-Deposit-Prediction
```

### 2. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### 3. Start the MLOps Pipeline

```bash
# Start all services
docker-compose up -d

# Monitor logs
docker-compose logs -f
```

### 4. Access Services

- **MLflow**: http://localhost:5000
- **FastAPI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Airflow**: http://localhost:8080
- Username: admin
- Password: Check with: docker logs mlops-airflow-webserver | grep -i "admin" | head -5

### 5. Trigger Training Pipeline

```bash
# Via Airflow UI or programmatically
curl -X POST "http://localhost:8080/api/v1/dags/bank_marketing_training_dag/dagRuns" \
  -H "Content-Type: application/json" \
  -d '{"dag_run_id": "manual_run_'$(date +%s)'"}'
```

## Machine Learning Pipeline

### 1. Data Ingestion
- Automated data download from Kaggle
- Data validation and quality checks
- Feature engineering and preprocessing

### 2. Model Training
- CatBoost classifier with hyperparameter tuning
- 5-fold cross-validation
- Automated model validation against thresholds

### 3. Model Validation
- Performance metrics: ROC-AUC, Precision, Recall, F1-Score
- Validation thresholds: min_roc_auc=0.85, min_accuracy=0.75
- SHAP explainability analysis

### 4. Drift Monitoring
- Statistical drift detection using Evidently AI
- Data quality monitoring
- Target drift detection
- Automated alerts and reporting

### 5. Model Deployment
- Conditional model promotion to production
- FastAPI serving with automatic reloading
- Health checks and monitoring

## Configuration

Key configurations in `config.yaml`:

```yaml
model:
  catboost:
    n_estimators: 5
    learning_rate: 0.1
    max_depth: 6

training:
  validation_thresholds:
    min_roc_auc: 0.85
    min_accuracy: 0.75

drift:
  thresholds:
    data_drift_p_value: 0.05
    target_drift_p_value: 0.05
```

## Monitoring and Alerts

### Drift Detection
- **Data Drift**: Statistical tests for feature distribution changes
- **Target Drift**: Monitor prediction target distribution shifts
- **Model Performance**: Track degradation over time

### Reports Generated
- Data drift analysis (`reports/data_drift_report.html`)
- Target drift analysis (`reports/target_drift_report.html`)
- Data quality assessment (`reports/data_quality_report.html`)
- SHAP feature importance (`reports/shap_summary.png`)

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Test DAGs
bash scripts/test_dags.sh

# Test API endpoints
python scripts/test_predictions.py
```

## Documentation

- **Architecture**: [docs/architecture.md](docs/architecture.md)
- **Dataset Details**: [docs/dataset.md](docs/dataset.md)
- **Data Dictionary**: [docs/data_dictionary.md](docs/data_dictionary.md)
- **Drift Strategy**: [docs/drift_plan.md](docs/drift_plan.md)

## External Resources

- **Dataset Source**: [Kaggle Playground Series S5E8](https://www.kaggle.com/competitions/playground-series-s5e8)
- **Original Dataset**: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Evidently AI Documentation**: https://evidentlyai.com/
- **MLflow Documentation**: https://mlflow.org/docs/
- **CatBoost Documentation**: https://catboost.ai/docs/

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper testing
4. Ensure pre-commit hooks pass
5. Submit a pull request

---