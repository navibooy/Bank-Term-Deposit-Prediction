# Bank Term Deposit Prediction - MLOps Architecture

## System Architecture Overview

This document provides a comprehensive overview of the MLOps architecture for the Bank Term Deposit Prediction system, detailing component interactions, data flows, and operational processes.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                           BANK TERM DEPOSIT PREDICTION - MLOPS ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAW DATA      │    │  PROCESSED DATA │    │ REFERENCE DATA  │    │   POSTGRESQL    │
│                 │    │                 │    │                 │    │                 │
│ • CSV Files     │    │ • Features      │    │ • Baseline      │    │ • Airflow DB    │
│ • Kaggle        │    │ • Train/Test    │    │ • Drift Ref     │    │ • MLflow DB     │
│ • External      │    │ • Validation    │    │ • Statistics    │    │ • Metadata      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       ▲                       ▲                       ▲
         │                       │                       │                       │
         │                       │                       │                       │
         ▼                       │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  TRAINING DAG   │────▶│ DEPLOYMENT DAG  │────▶│   DRIFT DAG     │──────────────┘
│                 │    │                 │    │                 │
│ • Data Ingest   │    │ • Model Valid   │    │ • Monitor Data  │
│ • Feature Eng   │    │ • Promotion     │    │ • Detect Drift  │
│ • Train Model   │    │ • Deploy        │    │ • Generate Rep  │
│ • Log MLflow    │    │ • Reload API    │    │ • Alert System │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       │                       ▼
┌─────────────────┐              │              ┌─────────────────┐
│     MLFLOW      │◀─────────────┘              │   EVIDENTLY     │
│                 │                             │                 │
│ • Experiments   │                             │ • Drift Reports │
│ • Models        │                             │ • Performance   │
│ • Artifacts     │                             │ • Data Quality  │
│ • Registry      │                             │ • Visualizations│
│ • Metrics       │                             └─────────────────┘
└─────────────────┘                                      │
         │                                               │
         │                                               ▼
         ▼                                      ┌─────────────────┐
┌─────────────────┐                            │   HTML REPORTS  │
│    FASTAPI      │                            │                 │
│                 │                            │ • Drift Reports │
│ • Model Serving │                            │ • Performance   │
│ • REST API      │                            │ • Data Quality  │
│ • Predictions   │                            │ • Alerts        │
│ • Health Checks │                            └─────────────────┘
│ • Admin Endpoints│
└─────────────────┘
         ▲
         │
         │
┌─────────────────┐
│      USERS      │
│                 │
│ • API Clients   │
│ • Web Interface │
│ • Applications  │
└─────────────────┘

LEGEND:
────▶  Data Flow / Process Trigger
━━━▶   Model Deployment
┅┅┅▶   Monitoring & Alerts
```

## Core Components

### 1. Data Layer

#### Raw Data Storage
- **Location**: `data/raw/`
- **Sources**: Kaggle datasets, external data sources
- **Format**: CSV, Parquet
- **Content**: Bank marketing campaign data with customer demographics

#### Processed Data Storage
- **Location**: `data/processed/`
- **Content**: Feature-engineered datasets, train/test splits
- **Processing**: Automated through Training DAG
- **Validation**: Data quality checks and schema validation

#### Reference Data
- **Location**: `data/reference/`
- **Purpose**: Baseline for drift detection
- **Generation**: Auto-generated from training data or manually curated
- **Usage**: Evidently AI drift analysis

### 2. Orchestration Layer (Apache Airflow)

#### Training DAG (`training_pipeline`)
**Schedule**: Daily (`@daily`)
**Trigger**: Scheduled execution
**Key Tasks**:
```python
data_ingestion → feature_engineering → model_training → model_validation → mlflow_logging
```
**Outputs**:
- Trained CatBoost model logged to MLflow
- Feature-engineered datasets
- Model performance metrics
- Triggers Deployment DAG upon successful completion

#### Deployment DAG (`deployment_pipeline`)
**Schedule**: Triggered by Training DAG
**Trigger**: `TriggerDagRunOperator` from Training DAG
**Key Tasks**:
```python
model_validation → performance_check → model_promotion → fastapi_reload
```
**Validation Thresholds**:
- Minimum ROC-AUC: 0.85
- Minimum Accuracy: 0.75
- Maximum CV Standard Deviation: 0.05

#### Drift DAG (`drift_detection_pipeline`)
**Schedule**: Hourly (`@hourly`)
**Trigger**: Scheduled execution
**Key Tasks**:
```python
generate_current_batch → detect_data_drift → detect_target_drift → generate_reports → alert_system
```
**Monitoring Scope**:
- Data distribution changes
- Target concept drift
- Feature importance shifts
- Data quality degradation

### 3. Model Management (MLflow)

#### Experiment Tracking
- **Experiment**: `bank-marketing-catboost`
- **Metrics**: ROC-AUC, Accuracy, F1-Score, Precision, Recall
- **Parameters**: CatBoost hyperparameters, feature engineering settings
- **Artifacts**: Models, SHAP plots, feature importance, confusion matrices

#### Model Registry
- **Registry Name**: `champion`
- **Stages**: `Staging` → `Production`
- **Promotion**: Automated through Deployment DAG
- **Versioning**: Automatic version management

#### Database Integration
- **Backend**: PostgreSQL
- **Schema**: MLflow tracking tables
- **Connection**: `postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow`

### 4. Model Serving (FastAPI)

#### API Endpoints
- **Health Check**: `GET /health`
- **Predictions**: `POST /predict`
- **Batch Predictions**: `POST /predict_batch`
- **Model Info**: `GET /model_info`
- **Admin Reload**: `POST /admin/reload`

#### Model Loading
- **Source**: MLflow Model Registry
- **Loading**: Startup + dynamic reload capability
- **Caching**: In-memory model caching for performance

#### Input Schema
```python
class PredictionInput(BaseModel):
    age: int
    balance: float
    duration: int
    campaign: int
    # ... additional banking features
```

### 5. Monitoring & Alerting

#### Evidently AI Integration
- **Data Drift Detection**: Statistical tests (KS, Chi-square)
- **Target Drift**: Label distribution analysis
- **Data Quality**: Missing values, duplicates, type mismatches
- **Performance Monitoring**: Model accuracy degradation

#### Report Generation
- **Format**: Interactive HTML reports
- **Storage**: `reports/` directory
- **Types**:
  - Data drift reports
  - Target drift reports
  - Data quality reports
  - Classification performance reports

#### Alerting System
- **Channels**: Logging, Email (configurable)
- **Thresholds**: Configurable p-values and performance metrics
- **Actions**: Model retraining triggers, notification workflows

### 6. Infrastructure (PostgreSQL)

#### Database Services
- **Airflow Metadata**: DAG runs, task instances, logs
- **MLflow Tracking**: Experiments, runs, metrics, parameters
- **Connection Pooling**: Optimized for concurrent access

## Component Interactions & Data Flow

### 1. Training Pipeline Flow

```
Raw Data → Data Ingestion → Feature Engineering → Model Training
    ↓
MLflow Logging ← Model Validation ← Cross-Validation ← Feature Selection
    ↓
Trigger Deployment DAG (if validation passes)
```

**Detailed Process**:
1. **Data Ingestion**: Load raw CSV data, perform initial validation
2. **Feature Engineering**: Apply transformations, create derived features
3. **Model Training**: Train CatBoost model with hyperparameter optimization
4. **Validation**: 5-fold cross-validation with performance thresholds
5. **MLflow Logging**: Log model, metrics, artifacts, and parameters
6. **Trigger**: Automatically trigger Deployment DAG if model meets criteria

### 2. Deployment Pipeline Flow

```
Model Validation → Performance Check → Model Promotion → FastAPI Reload
    ↓                    ↓                 ↓              ↓
Threshold Check →   Registry Stage →  Production →   Live Serving
```

**Detailed Process**:
1. **Model Validation**: Retrieve latest model from MLflow, validate metrics
2. **Performance Check**: Compare against production thresholds
3. **Model Promotion**: Move model from Staging to Production in registry
4. **FastAPI Reload**: Trigger API reload to serve new model
5. **Health Check**: Verify API is serving new model correctly

### 3. Drift Monitoring Flow

```
Reference Data + Current Batch → Evidently Analysis → Report Generation → Alerting
    ↓                              ↓                    ↓               ↓
Baseline Stats →              Statistical Tests →   HTML Reports →  Notifications
```

**Detailed Process**:
1. **Data Generation**: Create current batch from recent data or simulated drift
2. **Statistical Analysis**: Compare distributions using KS tests, Chi-square
3. **Report Creation**: Generate interactive HTML reports with visualizations
4. **Alerting**: Send notifications if drift exceeds configured thresholds
5. **Action Triggers**: Optionally trigger model retraining workflows

### 4. Model Serving Flow

```
User Request → FastAPI → Model Loading → Feature Transform → Prediction → Response
    ↓            ↓           ↓              ↓                 ↓           ↓
API Call →   Validation → MLflow Load →  Feature Eng →   CatBoost →  JSON Response
```

## Configuration Management

### Environment Variables
```yaml
# MLflow Configuration
MLFLOW_TRACKING_URI: http://mlflow:5000
MLFLOW_REGISTRY_URI: http://mlflow:5000
MLFLOW_EXPERIMENT_NAME: bank-marketing-catboost

# Model Configuration
MODEL_REGISTRY_NAME: champion
MODEL_REGISTRY_STAGE: Production

# Database Configuration
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres:5432/airflow
```

### Configuration File (`config.yaml`)
- **Model Parameters**: CatBoost hyperparameters, validation thresholds
- **Data Paths**: Raw, processed, reference data locations
- **Scheduling**: DAG schedules, retry policies
- **Monitoring**: Drift thresholds, alert configurations
- **API Settings**: Host, port, title, description

## Deployment Architecture

### Docker Services

#### Service Dependencies
```
postgres (base) → mlflow → airflow-webserver → fastapi
                     ↓
                 evidently reports
```

#### Network Configuration
- **Network**: `mlops-network` (bridge)
- **Internal Communication**: Service discovery via container names
- **External Access**: Exposed ports for web interfaces

#### Volume Management
- **Persistent Data**: PostgreSQL data, MLflow artifacts
- **Shared Storage**: Models, data, reports accessible across services
- **Log Storage**: Airflow logs, application logs

### Health Checks & Monitoring

#### Service Health Checks
- **PostgreSQL**: `pg_isready` command
- **MLflow**: HTTP health endpoint
- **FastAPI**: `/health` endpoint check
- **Airflow**: Webserver availability check

#### Monitoring Metrics
- **System**: CPU, Memory, Disk usage
- **Application**: Request latency, error rates
- **ML**: Model accuracy, prediction latency, drift metrics

## Security Considerations

### Authentication & Authorization
- **Airflow**: Admin user authentication
- **Database**: User-based access control
- **API**: Optional JWT token authentication (configurable)

### Data Security
- **Encryption**: Database connections use SSL/TLS
- **Secrets Management**: Environment variables for sensitive data
- **Access Control**: Role-based permissions for services

### Network Security
- **Internal Network**: Services communicate over private Docker network
- **Firewall**: Only necessary ports exposed externally
- **CORS**: Configured for FastAPI endpoints

## Scalability & Performance

### Horizontal Scaling
- **FastAPI**: Multiple replicas behind load balancer
- **Database**: Read replicas for MLflow tracking
- **Compute**: Resource limits and requests configured

### Performance Optimization
- **Model Caching**: In-memory model storage in FastAPI
- **Database Optimization**: Connection pooling, query optimization
- **Parallel Processing**: Concurrent task execution in Airflow

### Resource Management
- **Memory Limits**: 8GB maximum per service
- **CPU Limits**: 4 cores maximum per service
- **Storage**: Persistent volumes for critical data

## Troubleshooting Guide

### Common Issues

#### Model Loading Failures
**Symptoms**: FastAPI 500 errors, model not found
**Solutions**:
- Check MLflow registry for model availability
- Verify model stage (Production) in registry
- Restart FastAPI service to reload model

#### Drift Detection False Positives
**Symptoms**: Frequent drift alerts, normal data patterns
**Solutions**:
- Adjust drift thresholds in `config.yaml`
- Update reference data with recent representative samples
- Review statistical test sensitivity settings

#### DAG Execution Failures
**Symptoms**: Tasks failing, dependency issues
**Solutions**:
- Check Airflow logs for detailed error messages
- Verify data availability and format
- Ensure service dependencies are healthy

### Monitoring Dashboards

#### Key Metrics to Monitor
- **Model Performance**: ROC-AUC, Accuracy trends
- **Data Quality**: Missing values, schema violations
- **System Health**: Service uptime, response times
- **Pipeline Success**: DAG run success rates

## Future Enhancements

### Planned Improvements
- **A/B Testing**: Champion/Challenger model comparison
- **Auto-Scaling**: Dynamic resource allocation based on load
- **Advanced Monitoring**: Prometheus/Grafana integration
- **Model Interpretability**: Enhanced SHAP analysis and reporting
- **Data Lineage**: Complete data provenance tracking

### Integration Opportunities
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Cloud Migration**: AWS/Azure deployment with managed services
- **Real-time Streaming**: Kafka integration for live data processing
- **Advanced ML**: AutoML for hyperparameter optimization

---

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Git
- 8GB+ RAM recommended

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd Bank-Term-Deposit-Prediction

# Start all services
docker-compose up -d

# Access services
# Airflow: http://localhost:8080
#   Username: admin
#   Password: Check with: docker logs mlops-airflow-webserver | grep -i "admin" | head -5
# MLflow: http://localhost:5000
# FastAPI: http://localhost:8000
```

### Initial Setup
1. Wait for all services to be healthy (check logs)
2. Access Airflow webserver and trigger training pipeline
3. Monitor MLflow for experiment tracking
4. Test FastAPI endpoints once model is deployed
5. Review drift detection reports in `reports/` directory

This architecture provides a robust, scalable MLOps solution for bank term deposit prediction with comprehensive monitoring, automated deployment, and drift detection capabilities.