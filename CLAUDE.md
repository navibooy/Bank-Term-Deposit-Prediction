# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLOps project for predicting bank term deposit subscriptions using CatBoost. The system implements a complete machine learning pipeline with automated training, drift detection, model deployment, and monitoring using Airflow DAGs, MLflow for model tracking, and FastAPI for serving predictions.

## Development Commands

### Environment Setup
- **Install dependencies**: `uv sync` (using UV package manager)
- **Activate virtual environment**: `uv run <command>` or manually activate `.venv`

### Code Quality
- **Lint code**: `uv run ruff check --fix`
- **Format code**: `uv run ruff format`
- **Sort imports**: `uv run isort --profile black .`
- **Run pre-commit hooks**: `pre-commit run --all-files`

### Testing
- **Run tests**: `uv run pytest tests/`
- **Run specific test**: `uv run pytest tests/test_transform.py`

### MLflow Operations
- **Start MLflow server**: `uv run mlflow server --host 0.0.0.0 --port 5000`
- **MLflow tracking URI**: `http://localhost:5000`
- **Experiment name**: `bank-marketing-catboost`

### FastAPI Development
- **Start API server**: `uv run python src/serve/app.py`
- **API URL**: `http://localhost:8000`
- **API documentation**: `http://localhost:8000/docs`

### Airflow Operations
- **Airflow configuration**: See `docker/docker-compose.yaml` for Docker setup
- **DAG files location**: `dags/` directory
- **Main DAGs**: `training_dag.py`, `drift_dag.py`, `deployment_dag.py`

## Architecture Overview

The project follows a modular MLOps architecture:

### Core Components
1. **Data Pipeline** (`src/data/`): Data ingestion and drift simulation
2. **Feature Engineering** (`src/features/`): Data transformation and preprocessing
3. **Model Training** (`src/models/`): CatBoost training and validation with MLflow integration
4. **Model Serving** (`src/serve/`): FastAPI application for predictions
5. **Deployment** (`src/deployment/`): Model promotion and deployment logic
6. **Monitoring** (`src/monitoring/`): Drift detection using Evidently AI

### Data Flow
1. **Training DAG**: Ingests data → transforms features → trains model → validates → logs to MLflow
2. **Deployment DAG**: Checks model metrics → promotes champion model if thresholds met
3. **Drift DAG**: Monitors data/concept drift → generates Evidently reports → alerts on drift detection
4. **FastAPI**: Loads model from MLflow → serves predictions via REST API

### Configuration Management
- **Central config**: `config.yaml` contains all pipeline configurations
- **Model hyperparameters**: CatBoost parameters in `config.yaml` under `model.catboost`
- **Drift detection**: Comprehensive drift scenarios and thresholds configured
- **MLflow settings**: Experiment tracking and artifact logging configuration

### Key Features
- **Model Registry**: MLflow model versioning and promotion workflow
- **Drift Detection**: Evidently AI integration for comprehensive monitoring
- **Automated Retraining**: Airflow DAGs orchestrate the ML pipeline
- **Performance Validation**: Configurable thresholds for model promotion
- **Docker Support**: Full containerization with docker-compose setup

## Important File Locations

### Configuration
- **Main config**: `config.yaml` - Central configuration for all components
- **Dependencies**: `pyproject.toml` - UV-managed Python dependencies
- **Pre-commit**: `.pre-commit-config.yaml` - Code quality hooks

### Data Directories
- **Raw data**: `data/raw/` - Original dataset files
- **Processed data**: `data/processed/` - Transformed training/test sets
- **Reference data**: `data/reference/` - Baseline data for drift detection
- **Current data**: `data/current/` - Live data batches for monitoring

### Model Artifacts
- **Trained models**: `models/` directory and MLflow artifact store
- **MLflow artifacts**: `mlartifacts/` and `mlruns/` directories
- **Reports**: `reports/` - Drift detection and model performance reports

### Development Guidelines

#### Model Development
- Use CatBoost as the primary algorithm (configured in `config.yaml`)
- All training runs logged to MLflow with comprehensive metrics
- Model promotion based on configurable validation thresholds
- Feature importance and SHAP plots automatically generated

#### Data Handling
- Follow the established data schema in the feature transformation pipeline
- Use the drift simulation capabilities for testing monitoring systems
- Maintain data versioning through the configured directory structure

#### API Development
- FastAPI service loads models from MLflow registry
- Update `PredictionInput` schema in `src/serve/app.py` to match your features
- Health checks and model metadata endpoints available

#### Monitoring and Alerting
- Evidently AI reports generated for data quality, drift, and performance
- Configurable alert thresholds in `config.yaml` under `drift` section
- Reports saved as HTML files in `reports/` directory

## Docker and Deployment

The project includes comprehensive Docker setup:
- **Airflow**: `docker/airflow.Dockerfile`
- **FastAPI**: `docker/fastapi.Dockerfile`
- **MLflow**: `docker/mlflow.Dockerfile`
- **Orchestration**: `docker/docker-compose.yaml`

Use `docker-compose up` to start the full stack including Airflow, MLflow, and PostgreSQL.