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
- **Airflow UI**: `http://localhost:8080` (admin/sb69CHbtsUAaPv5G)

### DAG Testing
- **Quick test all DAGs**: `./scripts/test_dags.sh`
- **Test DAG syntax**: `./scripts/test_dags.sh syntax`
- **Test specific task**: `./scripts/test_dags.sh test-task training_pipeline data_ingestion 2025-08-28`
- **Test individual tasks**: `uv run python scripts/test_tasks.py pipeline`
- **Run DAG unit tests**: `uv run pytest tests/test_dags.py -v`

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

## DAG Testing Guide

The project includes comprehensive testing capabilities for Airflow DAGs to ensure reliability and catch issues early in development.

### Testing Approaches

#### 1. Quick Syntax and Import Testing
```bash
# Test all DAG files for syntax errors
./scripts/test_dags.sh syntax

# Test DAG imports locally
./scripts/test_dags.sh imports

# Test DAG imports in Docker container
./scripts/test_dags.sh container
```

#### 2. Individual Task Testing
```bash
# Test specific tasks outside Airflow
uv run python scripts/test_tasks.py ingestion
uv run python scripts/test_tasks.py transformation
uv run python scripts/test_tasks.py training
uv run python scripts/test_tasks.py validation

# Test complete pipeline
uv run python scripts/test_tasks.py pipeline
```

#### 3. Airflow Container Testing
```bash
# Test DAG structure in container
./scripts/test_dags.sh structure training_pipeline

# Test individual task in Airflow
./scripts/test_dags.sh test-task training_pipeline data_ingestion 2025-08-28

# Run all container tests
./scripts/test_dags.sh container
```

#### 4. Unit Testing with Pytest
```bash
# Run all DAG unit tests
uv run pytest tests/test_dags.py -v

# Run specific test categories
uv run pytest tests/test_dags.py::TestDAGIntegrity -v
uv run pytest tests/test_dags.py::TestDAGTasks -v

# Run integration tests (requires Airflow)
uv run pytest tests/test_dags.py -m integration -v
```

### Testing Utilities

#### DAG Testing Script (`scripts/test_dags.sh`)
Comprehensive bash script with multiple testing commands:
- **Syntax validation**: Checks Python syntax of all DAG files
- **Import testing**: Tests DAG imports in both local and Docker environments
- **Container testing**: Validates DAGs within the Airflow container
- **Task testing**: Executes individual tasks with test data
- **Structure validation**: Verifies DAG dependencies and configuration

#### Task Testing Utility (`scripts/test_tasks.py`)
Python script for testing individual tasks outside Airflow:
- **Mock contexts**: Creates realistic Airflow task contexts
- **XCom simulation**: Mocks inter-task communication
- **Pipeline testing**: Tests full pipeline workflows
- **Results saving**: Saves test results to JSON for analysis

#### Unit Test Suite (`tests/test_dags.py`)
Comprehensive pytest-based test suite:
- **DAG integrity**: Tests DAG structure, dependencies, and configuration
- **Task functionality**: Tests individual task functions with mocked data
- **Error handling**: Validates error scenarios and edge cases
- **Integration testing**: Tests DAG loading in DagBag (requires Airflow)

### Common Testing Workflows

#### Development Testing
```bash
# 1. Test syntax after making changes
./scripts/test_dags.sh syntax

# 2. Test specific task logic
uv run python scripts/test_tasks.py training

# 3. Run unit tests
uv run pytest tests/test_dags.py::TestDAGTasks::test_model_training_task_success -v
```

#### Pre-deployment Testing
```bash
# 1. Run all tests
./scripts/test_dags.sh all

# 2. Test in container environment
docker-compose up -d
./scripts/test_dags.sh container

# 3. Run integration tests
uv run pytest tests/test_dags.py -v
```

#### Debugging Failed DAGs
```bash
# 1. Check import errors
./scripts/test_dags.sh imports

# 2. Test specific failing task
./scripts/test_dags.sh test-task training_pipeline data_ingestion 2025-08-28

# 3. Run detailed task testing
uv run python scripts/test_tasks.py ingestion --verbose
```

### Troubleshooting Common Issues

#### Import Errors (`No module named 'src'`)
- **Cause**: Python path not configured correctly
- **Solution**: Ensure all DAG files have proper path setup (fixed in current DAGs)
- **Test**: Run `./scripts/test_dags.sh imports` to verify

#### Task Failures
- **Cause**: Missing data files, configuration issues, or dependency problems
- **Solution**: Use task testing utility to debug with mock data
- **Test**: Run `uv run python scripts/test_tasks.py <task_name> --verbose`

#### Missing Dependencies (`No module named 'catboost'`)
- **Cause**: ML dependencies not installed in Airflow container
- **Solution**: Rebuild container with updated dependencies
- **Commands**:
  ```bash
  docker-compose down
  docker-compose up --build -d
  ```
- **Test**: Run `./scripts/test_dags.sh dependencies` to verify

#### Container Connection Issues
- **Cause**: Docker containers not running or network issues
- **Solution**: Ensure `docker-compose up -d` is running successfully
- **Test**: Run `./scripts/test_dags.sh container` to verify

#### MLflow/Database Connectivity
- **Cause**: Services not started or configuration issues
- **Solution**: Check service health and configuration
- **Test**: Run `./scripts/test_dags.sh mlflow` to verify connectivity

### Testing Configuration

#### Test Data Setup
The testing utilities automatically create mock data for testing, but for more realistic tests:
1. Ensure sample data files exist in `data/raw/`
2. Run `uv run python src/data/ingest.py` to create processed data
3. Configure test-specific data paths if needed

#### Environment Variables
Tests respect the same environment variables as the main application:
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `PYTHONPATH`: Python module search paths
- Configuration loaded from `config.yaml`

#### Continuous Integration
The testing utilities are designed to work in CI/CD environments:
```bash
# CI pipeline testing commands
./scripts/test_dags.sh syntax
uv run pytest tests/test_dags.py --tb=short
uv run python scripts/test_tasks.py pipeline --save
```