# Containerization Architecture

## Overview

The Bank Term Deposit Prediction MLOps pipeline is fully containerized using Docker and orchestrated with Docker Compose. The system consists of four main services that work together to provide a complete machine learning operations platform.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Network (mlops-network)              │
├─────────────┬─────────────┬─────────────────┬─────────────────┤
│   Airflow   │   MLflow    │     FastAPI     │   PostgreSQL    │
│  (Port 8080)│ (Port 5000) │   (Port 8000)   │   (Internal)    │
│             │             │                 │                 │
│ ┌─────────┐ │ ┌─────────┐ │ ┌─────────────┐ │ ┌─────────────┐ │
│ │ DAGs    │ │ │Registry │ │ │Model Serving│ │ │Metadata     │ │
│ │Training │ │ │Tracking │ │ │Predictions  │ │ │Storage      │ │
│ │Drift    │ │ │Artifacts│ │ │Health Check │ │ │Users & Roles│ │
│ └─────────┘ │ └─────────┘ │ └─────────────┘ │ └─────────────┘ │
└─────────────┴─────────────┴─────────────────┴─────────────────┘
```

## Service Containerization Details

### 1. PostgreSQL Database
- **Image**: `postgres:15`
- **Purpose**: Backend database for MLflow and Airflow metadata
- **Configuration**:
  - Creates separate databases: `mlflow` and `airflow`
  - Initializes users and permissions via `init-db.sql`
  - Persistent storage with named volume `postgres_data`
  - Health check with `pg_isready`

### 2. MLflow Tracking Server
- **Base Image**: `python:3.12-slim`
- **Dockerfile**: `docker/mlflow.Dockerfile`
- **Purpose**: Model tracking, registry, and experiment management
- **Key Features**:
  - PostgreSQL backend store for metadata
  - Artifact storage in `/mlruns` directory
  - Built-in health endpoint
  - Non-root user execution (`mlflow`)
  - Version pinned to `mlflow==3.3.1`

**Dependencies Installed**:
```dockerfile
mlflow==3.3.1
psycopg2-binary
boto3
pandas==2.3.2
numpy==2.2.6
scikit-learn==1.7.1
pyyaml
```

### 3. Airflow Orchestration
- **Base Image**: `python:3.12-slim`
- **Dockerfile**: `docker/airflow.Dockerfile`
- **Purpose**: Workflow orchestration for training and monitoring DAGs
- **Configuration**:
  - LocalExecutor for single-node execution
  - PostgreSQL backend for metadata
  - Built-in admin user creation
  - Volume mounts for DAGs and source code

**Key ML Dependencies**:
```dockerfile
apache-airflow[postgres]==3.0.4
catboost==1.2.8
evidently==0.7.11
mlflow==3.3.1
scikit-learn==1.7.1
```

### 4. FastAPI Model Serving
- **Base Image**: `python:3.12-slim`
- **Dockerfile**: `docker/fastapi.Dockerfile`
- **Purpose**: REST API for model predictions and health monitoring
- **Features**:
  - Uvicorn ASGI server
  - MLflow model registry integration
  - Health check endpoint
  - Non-root user execution (`fastapi`)

**Runtime Dependencies**:
```dockerfile
fastapi==0.116.1
uvicorn[standard]
mlflow==3.3.1
catboost==1.2.8
pandas==2.3.2
numpy==2.2.6
```

## Volume Management

### Persistent Data Volumes
- **`postgres_data`**: Database storage
- **`mlflow_data`**: MLflow metadata persistence
- **`airflow_logs`**: Airflow execution logs

### Bind Mounts
- **`./data`**: Training and inference datasets
- **`./models`**: Local model artifacts
- **`./mlruns`**: MLflow experiment runs
- **`./mlartifacts`**: MLflow artifacts
- **`./dags`**: Airflow DAG definitions
- **`./src`**: Source code for ML pipeline
- **`./config.yaml`**: Application configuration

## Network Configuration

### Service Discovery
All services communicate through the `mlops-network` bridge network:
- Internal DNS resolution by service name (e.g., `mlflow:5000`)
- Isolated network namespace from host
- Only necessary ports exposed to host system

### Port Mappings
| Service | Internal Port | External Port | Purpose |
|---------|---------------|---------------|---------|
| Airflow | 8080 | 8080 | Web UI |
| MLflow | 5000 | 5000 | Tracking Server |
| FastAPI | 8000 | 8000 | API Endpoints |
| PostgreSQL | 5432 | - | Database (internal only) |

## Health Checks & Dependencies

### Health Check Configuration
Each service implements health checks:
- **MLflow**: `curl -f http://localhost:5000/health`
- **Airflow**: `curl -f http://localhost:8080/health`
- **FastAPI**: `curl -f http://localhost:8000/health`
- **PostgreSQL**: `pg_isready -U postgres`

### Dependency Chain
```
PostgreSQL → MLflow → FastAPI
     ↓
  Airflow
```

Services wait for their dependencies to be healthy before starting, ensuring proper initialization order.

## Security Considerations

### User Isolation
- Each service runs as a dedicated non-root user
- Proper file ownership and permissions
- Limited system access within containers

### Current Limitations
- Default passwords in use (development setup)
- No TLS/SSL encryption between services
- No authentication on MLflow and FastAPI endpoints

### Production Hardening Needed
- Environment-specific credentials
- Secret management (Docker Secrets/External)
- Service authentication and authorization
- Network segmentation and firewall rules

## Development Workflow

### Local Development
```bash
# Start core services only
docker-compose up postgres mlflow -d

# Develop FastAPI locally
source .venv/bin/activate
python src/serve/app.py
```

### Full Pipeline Testing
```bash
# Build and start all services
docker-compose up --build -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f [service-name]
```

### Service Management
```bash
# Restart specific service
docker-compose restart fastapi

# Scale services (if configured)
docker-compose up --scale fastapi=3

# Stop all services
docker-compose down

# Remove volumes (clean slate)
docker-compose down -v
```

## Monitoring & Debugging

### Log Access
```bash
# Real-time logs
docker-compose logs -f

# Service-specific logs
docker-compose logs airflow-webserver
docker-compose logs mlflow
docker-compose logs fastapi

# Container shell access
docker-compose exec fastapi bash
```

### Health Monitoring
```bash
# Check all service health
curl http://localhost:8000/health
curl http://localhost:5000/health
curl http://localhost:8080/health

# Service status overview
docker-compose ps
```

## Resource Usage

### Memory Requirements
- **PostgreSQL**: ~100-200MB
- **MLflow**: ~200-400MB
- **Airflow**: ~400-800MB
- **FastAPI**: ~100-300MB

### Storage Requirements
- **Base Images**: ~2GB total
- **Persistent Volumes**: Varies by data size
- **Logs**: Grows with usage

### CPU Usage
- Generally low during idle
- Peaks during model training (Airflow)
- Consistent low load for serving (FastAPI)

## Future Improvements

### Scalability Enhancements
- CeleryExecutor for Airflow horizontal scaling
- FastAPI replica scaling with load balancer
- External PostgreSQL cluster
- Shared artifact storage (S3/MinIO)

### Operational Improvements
- Multi-stage Dockerfile builds for smaller images
- Resource limits and reservations
- Prometheus metrics collection
- Centralized logging (ELK/Grafana)

### Security Enhancements
- Secret management integration
- Service mesh (Istio/Linkerd)
- Network policies and segmentation
- Image vulnerability scanning