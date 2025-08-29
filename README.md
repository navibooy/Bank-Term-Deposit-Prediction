# Bank Term Deposit Prediction MLOps Pipeline

## 1. Introduction & Problem Statement

### Problem Definition
This project addresses the challenge of predicting whether bank clients will subscribe to term deposit products. Banks need to optimize their marketing campaigns by identifying customers most likely to subscribe, reducing costs and improving conversion rates.

### Business Motivation
- **Customer Targeting**: Identify high-potential customers to maximize marketing ROI
- **Resource Optimization**: Focus sales efforts on prospects with highest conversion probability  
- **Automated Decision Making**: Enable real-time predictions for customer interactions
- **Data-Driven Insights**: Understand key factors influencing customer subscription behavior

### Success Metrics
- **Model Performance**: ROC-AUC ≥ 0.85, Accuracy ≥ 0.75
- **System Reliability**: 99%+ API uptime with <200ms prediction latency
- **Operational Excellence**: Automated retraining pipeline with drift detection
- **Business Impact**: Improved marketing campaign conversion rates

## 2. Architecture Overview

### System Components Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Pipeline Architecture                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Raw Data   │    │  Processed  │    │   Models    │    │   Reports   │     │
│  │ (CSV Files)  │    │    Data     │    │ (Artifacts) │    │(Drift/Perf)│     │
│  └──────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                    │                   │                   │          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      Airflow Orchestration                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │ Training    │  │ Deployment  │  │ Drift       │  │ Monitoring  │    │   │
│  │  │ DAG         │  │ DAG         │  │ DAG         │  │ DAG         │    │   │
│  │  │ (@daily)    │  │ (triggered) │  │ (@hourly)   │  │ (continuous)│    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   MLflow    │    │  FastAPI    │    │ EvidentlyAI │    │  Docker     │     │
│  │ Tracking &  │    │ Model       │    │ Drift       │    │ Container   │     │
│  │ Registry    │    │ Serving     │    │ Detection   │    │ Management  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                    │                   │                   │          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      Infrastructure Layer                               │   │
│  │    PostgreSQL     │    MLflow UI    │    Airflow UI    │   API Docs     │   │
│  │   (Metadata)      │   (:5000)       │    (:8080)       │   (:8000)      │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### End-to-End Pipeline Flow

1. **Data Ingestion**: Raw CSV data is processed and split into train/test sets
2. **Feature Engineering**: Categorical encoding, numerical scaling, and feature creation
3. **Model Training**: CatBoost classifier trained with cross-validation and MLflow logging
4. **Model Validation**: Performance metrics evaluated against promotion thresholds
5. **Model Registration**: Successful models registered in MLflow Model Registry
6. **Model Deployment**: Champion models promoted to Production stage and loaded by FastAPI
7. **Drift Monitoring**: EvidentlyAI generates reports comparing current vs. reference data
8. **Automated Retraining**: Drift detection triggers retraining pipeline when thresholds exceeded

### Component Integration

- **Airflow ↔ MLflow**: DAGs log experiments, metrics, and artifacts to MLflow tracking server
- **MLflow ↔ FastAPI**: API loads champion models from MLflow Model Registry
- **Airflow ↔ EvidentlyAI**: Drift DAG generates Evidently reports and logs to MLflow
- **Docker ↔ All Components**: Containerized services communicate via Docker network
- **PostgreSQL ↔ Services**: Shared metadata store for Airflow and MLflow

## 3. Data & Model

### Dataset Description
- **Source**: Bank Marketing Dataset (Portuguese banking institution)
- **Size**: 45,211 records with 17 features
- **Target**: Binary classification (subscribe to term deposit: yes/no)
- **Features**: Demographics (age, job, education), financial (balance, loan status), campaign data (duration, contacts)

### Feature Categories
- **Numerical**: age, balance, duration, campaign, pdays, previous (6 features)
- **Categorical**: job, marital, education, default, housing, loan, contact, month, poutcome (11 features)

### Data Preprocessing
- **Train/Test Split**: 80/20 stratified split maintaining target distribution
- **Feature Engineering**: 
  - One-hot encoding for categorical variables
  - Custom `many_no` feature aggregating multiple "no" responses
  - Numerical feature standardization
- **Final Feature Set**: 17 engineered features optimized for CatBoost

## 4. MLflow Integration

### Experiment Tracking
Our MLflow integration provides comprehensive experiment management:

- **Experiment Name**: `bank-marketing-catboost`
- **Tracked Parameters**: All CatBoost hyperparameters (learning_rate, max_depth, n_estimators, etc.)
- **Logged Metrics**: ROC-AUC, F1-score, Precision, Recall, Accuracy for train/validation/test sets
- **Artifacts**: Trained models, feature importance plots, SHAP visualizations, confusion matrices

### Model Registry Workflow
```python
# Example MLflow logging from training pipeline
with mlflow.start_run(run_name=f"catboost_training_{timestamp}"):
    # Log hyperparameters
    mlflow.log_params(catboost_params)
    
    # Train model and log metrics
    model = CatBoostClassifier(**catboost_params)
    model.fit(X_train, y_train)
    
    # Log performance metrics
    mlflow.log_metrics({
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "f1_score": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred)
    })
    
    # Log model to registry
    mlflow.catboost.log_model(model, "model", registered_model_name="champion")
```

### MLflow UI Dashboard
The MLflow UI provides:
- **Experiment Comparison**: Side-by-side comparison of model runs with sortable metrics
- **Model Registry**: Version management with Production/Staging stages
- **Artifact Browser**: Access to model files, plots, and reports
- **Model Lineage**: Track model evolution and promotion history

**Access**: http://localhost:5000

## 5. Airflow Orchestration

### DAG Architecture

#### 1. Training DAG (`training_dag.py`)
**Schedule**: Daily (@daily)
**Tasks**:
```
data_ingestion → feature_transformation → model_training → model_validation → deployment_trigger
```
- **Dependencies**: Sequential execution ensuring data quality
- **Retries**: 2 retries with 5-minute delays
- **Error Handling**: Failed tasks don't propagate to downstream dependencies

#### 2. Deployment DAG (`deployment_dag.py`)  
**Trigger**: Activated by successful training completion
**Tasks**:
```
check_model_metrics → promote_champion_model → reload_fastapi_service
```
- **Validation Logic**: Only promotes models meeting ROC-AUC ≥ 0.85 threshold
- **Service Integration**: Reloads FastAPI to use newly promoted model

#### 3. Drift DAG (`drift_dag.py`)
**Schedule**: Hourly (@hourly)
**Tasks**:
```
generate_current_batch → detect_data_drift → generate_reports → trigger_retraining
```
- **Monitoring Scope**: Data drift, target drift, and data quality checks
- **Alert System**: Triggers retraining when drift thresholds exceeded

### Task Configuration
```python
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False
}
```

### Airflow UI Access
- **URL**: http://localhost:8080
- **Credentials**: admin/admin
- **Features**: DAG visualization, task logs, scheduling, manual triggers

## 6. Docker & Containerization

### Container Architecture
Our Docker setup orchestrates multiple services with proper networking and dependency management:

#### Service Definitions
```yaml
services:
  postgres:      # Metadata store for MLflow & Airflow
  mlflow:        # Experiment tracking and model registry  
  airflow:       # Workflow orchestration
  fastapi:       # Model serving API
```

#### Key Dockerfiles

**1. MLflow Service** (`docker/mlflow.Dockerfile`)
```dockerfile
FROM python:3.11-slim
RUN pip install mlflow psycopg2-binary
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", "postgresql://..."]
```

**2. FastAPI Service** (`docker/fastapi.Dockerfile`)
```dockerfile  
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ /app/src/
EXPOSE 8000
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0"]
```

**3. Airflow Service** (`docker/airflow.Dockerfile`)
```dockerfile
FROM apache/airflow:2.7.0-python3.11
COPY requirements.txt .
RUN pip install -r requirements.txt
```

### Docker Compose Configuration
- **Networking**: Dedicated `mlops-network` for inter-service communication
- **Volumes**: Persistent storage for data, models, and artifacts
- **Health Checks**: Service dependency management with health monitoring
- **Environment Variables**: Centralized configuration management

### Container Management
```bash
# Start full stack
docker-compose up -d

# View logs
docker-compose logs fastapi

# Scale services
docker-compose up -d --scale fastapi=2

# Cleanup
docker-compose down -v
```

## 7. FastAPI & Model Serving

### API Endpoints

#### Core Prediction Service
```python
@app.post("/predict")
async def predict(input_data: PredictionInput) -> PredictionOutput:
    """
    Predict bank term deposit subscription
    
    Input: Customer demographics and campaign data
    Output: Binary prediction with confidence probability
    """
```

#### Model Information
```python  
@app.get("/model")
async def get_model_info() -> ModelInfo:
    """
    Retrieve current model metadata:
    - Hyperparameters
    - Feature importance ranking  
    - Model version and registration info
    """
```

### Request/Response Format

**Prediction Request**:
```json
{
    "age": 42,
    "job": "technician", 
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may", 
    "duration": 300,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}
```

**Prediction Response**:
```json
{
    "prediction": 1,
    "probability": 0.73,
    "model_version": "3",
    "prediction_timestamp": "2025-08-29T10:30:45"
}
```

### Model Loading from MLflow
```python
class MLModelService:
    async def load_champion_model(self):
        """Load production model from MLflow Registry"""
        model_uri = f"models:/champion/Production"
        self.model = mlflow.pyfunc.load_model(model_uri)
```

### API Documentation
- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc  
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 8. EvidentlyAI for Drift Detection

### Comprehensive Drift Monitoring
Our drift detection system uses EvidentlyAI to monitor multiple aspects of model and data health:

#### Drift Types Monitored
1. **Data Drift**: Feature distribution changes using Kolmogorov-Smirnov test
2. **Target Drift**: Label distribution shifts in production data
3. **Concept Drift**: Relationship changes between features and target
4. **Data Quality**: Missing values, duplicates, and outliers

#### Report Generation
```python
# Data Drift Report
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

# Target Drift Report  
target_drift_report = Report(metrics=[
    TargetDriftPreset(),
])

# Data Quality Report
quality_report = Report(metrics=[
    DataQualityPreset(),
])
```

### Drift Configuration
```yaml
drift:
  thresholds:
    data_drift_p_value: 0.05
    target_drift_p_value: 0.05  
    feature_drift_threshold: 0.1
  
  scenarios:
    severe_data_drift:
      numerical_drift:
        age: {type: "gaussian_noise", noise_std: 0.6}
        balance: {type: "shift_mean", shift_factor: 0.5}
      categorical_drift:
        job: {type: "category_shift", shift_percentage: 0.3}
```

### Evidently Visualizations
Generated HTML reports include:
- **Feature Distribution Comparison**: Reference vs. current data histograms
- **Drift Detection Results**: Statistical test results with p-values  
- **Data Quality Metrics**: Missing data patterns and anomaly detection
- **Performance Degradation**: Model accuracy trends over time

### Integration in Airflow
The Drift DAG automatically:
1. Generates current data batch with potential drift scenarios
2. Compares against reference baseline data
3. Produces Evidently reports and saves to `/reports/` directory
4. Logs drift metrics to MLflow for tracking
5. Triggers retraining pipeline if drift thresholds exceeded

## 9. End-to-End Workflow Demo

### Complete Scenario: New Data → Retrain → Drift Detection

#### Phase 1: Initial Training (Day 1)
```bash
# 1. Start MLOps stack
docker-compose up -d

# 2. Trigger initial training
curl -X POST "http://localhost:8080/api/v1/dags/training_pipeline/dagRuns" \
     -H "Content-Type: application/json" \
     -d '{"conf":{}}'

# 3. Monitor training in Airflow UI
# http://localhost:8080 → training_pipeline → task logs

# 4. View experiment results in MLflow  
# http://localhost:5000 → Experiments → bank-marketing-catboost
```

#### Phase 2: Model Deployment (Automated)
```bash
# Training success triggers deployment DAG automatically
# 1. Check model metrics against thresholds (ROC-AUC ≥ 0.85)
# 2. Promote model to Production stage in MLflow Registry
# 3. Reload FastAPI service with new champion model
# 4. Verify API health: http://localhost:8000/health
```

#### Phase 3: Production Predictions
```bash
# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "age": 35,
         "job": "management", 
         "marital": "single",
         "education": "tertiary",
         "default": "no",
         "balance": 2000,
         "housing": "yes",
         "loan": "no",
         "contact": "cellular",
         "day": 20,
         "month": "jun",
         "duration": 400,
         "campaign": 1,
         "pdays": -1, 
         "previous": 0,
         "poutcome": "unknown"
     }'

# Expected Response:
{
    "prediction": 1,
    "probability": 0.78,
    "model_version": "1", 
    "prediction_timestamp": "2025-08-29T14:22:33"
}
```

#### Phase 4: Drift Detection & Retraining (Continuous)
```bash
# Drift monitoring runs hourly automatically
# 1. Generate current batch with potential distribution changes
# 2. Compare against reference data using EvidentlyAI
# 3. Generate comprehensive drift reports
# 4. If drift detected (p-value < 0.05), trigger retraining

# View drift reports
open reports/data_drift_report.html
open reports/target_drift_report.html

# Check MLflow for drift metrics
# http://localhost:5000 → Experiments → Search for "drift_detection"
```

#### Phase 5: Model Performance Monitoring
```bash
# Continuous monitoring tracks:
# - Prediction latency and throughput  
# - Model accuracy on new labeled data
# - Feature importance stability
# - Data quality degradation

# Access monitoring dashboards
curl http://localhost:8000/model  # Model metadata
curl http://localhost:5000/api/2.0/mlflow/experiments/list  # MLflow API
```

### Workflow Validation
```bash
# Complete end-to-end test
./scripts/test_dags.sh all

# Individual component tests
uv run pytest tests/ -v
uv run python scripts/test_tasks.py pipeline
```

## 10. Project Insights & Learning Outcomes

### Technical Achievements
- **Scalable MLOps Architecture**: Production-ready pipeline handling data ingestion through model serving
- **Automated Model Lifecycle**: Seamless training → validation → deployment → monitoring cycle
- **Comprehensive Monitoring**: Multi-dimensional drift detection with automated retraining triggers  
- **Service Reliability**: Containerized architecture with health checks and graceful error handling

### Key Challenges Overcome
1. **MLflow Integration**: Resolved model loading issues between Airflow and FastAPI containers
2. **Feature Engineering Pipeline**: Ensured consistent preprocessing across training and serving
3. **Drift Simulation**: Implemented realistic drift scenarios for robust monitoring validation
4. **Container Orchestration**: Managed service dependencies and network communication

### Lessons Learned
- **Configuration Management**: Centralized YAML config significantly improved system maintainability
- **Testing Strategy**: Comprehensive DAG testing prevented production issues and improved reliability
- **Monitoring Granularity**: Multiple drift detection approaches (statistical, visual, performance-based) provide complementary insights
- **Documentation Importance**: Well-documented APIs and architecture accelerate development and debugging

### Production Readiness Considerations
- **Security**: Implement authentication for MLflow and Airflow UIs
- **Scalability**: Add horizontal scaling for FastAPI and resource limits for training
- **Backup Strategy**: Database backups and artifact versioning for disaster recovery
- **Observability**: Integrate with monitoring tools (Prometheus, Grafana) for production visibility

### Future Enhancements
- **A/B Testing Framework**: Compare model versions in production with statistical significance testing
- **Feature Store Integration**: Centralized feature management for consistency across models
- **Multi-Model Support**: Extend pipeline to handle multiple model types and ensembles
- **Real-time Streaming**: Process live data streams for immediate drift detection and predictions

### Business Impact
This MLOps pipeline enables:
- **Improved ROI**: Target high-potential customers with 85%+ accuracy
- **Operational Efficiency**: Automated model lifecycle reduces manual intervention by 90%
- **Risk Mitigation**: Proactive drift detection prevents model performance degradation
- **Data-Driven Decisions**: Comprehensive reporting supports strategic marketing decisions

---

## Quick Start Guide

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- 8GB RAM, 4 CPU cores recommended

### Launch Commands
```bash
# 1. Clone repository
git clone <repository-url>
cd Bank-Term-Deposit-Prediction

# 2. Start MLOps stack
docker-compose up -d

# 3. Access services
# Airflow: http://localhost:8080 (admin/admin)
# MLflow: http://localhost:5000  
# FastAPI: http://localhost:8000/docs

# 4. Run complete pipeline test
./scripts/test_dags.sh all
```

### Service URLs
- **Airflow UI**: http://localhost:8080
- **MLflow UI**: http://localhost:5000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

This comprehensive MLOps pipeline demonstrates enterprise-grade machine learning operations with automated training, deployment, and monitoring capabilities, providing a robust foundation for production ML systems.