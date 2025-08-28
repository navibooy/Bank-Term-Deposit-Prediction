# Docker Infrastructure for Bank Term Deposit MLOps Pipeline

This directory contains the Docker infrastructure for running the complete MLOps pipeline in containerized environments.

## Architecture Overview

The system consists of the following services:

- **PostgreSQL**: Backend database for MLflow and Airflow metadata
- **MLflow**: Model tracking and registry service
- **Airflow**: Workflow orchestration (webserver + scheduler)
- **FastAPI**: Model serving application

## Services Communication

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Airflow   │───▶│   MLflow    │◀───│   FastAPI   │
│  (DAGs)     │    │ (Registry)  │    │ (Serving)   │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────▼───────────────────┘
                   ┌─────────────┐
                   │ PostgreSQL  │
                   │ (Backend)   │
                   └─────────────┘
```

## Quick Start

### 1. Build and Start All Services

```bash
# Start the complete MLOps stack
docker-compose up --build -d

# View service logs
docker-compose logs -f [service-name]

# Check service status
docker-compose ps
```

### 2. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow Webserver | http://localhost:8080 | admin/admin |
| MLflow Tracking | http://localhost:5000 | No auth |
| FastAPI | http://localhost:8000 | No auth |
| FastAPI Docs | http://localhost:8000/docs | No auth |

### 3. Verify Services

```bash
# Check health endpoints
curl http://localhost:8000/health  # FastAPI
curl http://localhost:5000/health  # MLflow
curl http://localhost:8080/health  # Airflow
```

## Service Configuration

### Environment Variables

**MLflow Configuration:**
- `MLFLOW_TRACKING_URI`: http://mlflow:5000
- `MLFLOW_BACKEND_STORE_URI`: postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow

**Airflow Configuration:**
- `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`: postgresql+psycopg2://airflow:airflow@postgres:5432/airflow
- `AIRFLOW__CORE__EXECUTOR`: LocalExecutor

**FastAPI Configuration:**
- `MLFLOW_TRACKING_URI`: http://mlflow:5000
- `PYTHONPATH`: /app

### Volume Mounts

**Data Persistence:**
- `./data:/app/data` - Training and inference data
- `./models:/app/models` - Local model artifacts
- `./mlruns:/mlruns` - MLflow experiment runs
- `./mlartifacts:/mlartifacts` - MLflow artifacts
- `./reports:/app/reports` - Drift monitoring reports

**Database Storage:**
- `postgres_data:/var/lib/postgresql/data` - PostgreSQL data
- `airflow_logs:/opt/airflow/logs` - Airflow execution logs

## Service Dependencies

**Startup Order:**
1. PostgreSQL (database backend)
2. MLflow (depends on PostgreSQL)
3. Airflow Webserver (depends on PostgreSQL + MLflow)
4. Airflow Scheduler (depends on Airflow Webserver)
5. FastAPI (depends on MLflow)

**Health Checks:**
- All services have built-in health checks
- Services wait for dependencies to be healthy before starting
- Automatic restart on failure with `unless-stopped` policy

## Development Workflow

### 1. Local Development

```bash
# Start only required services for development
docker-compose up postgres mlflow -d

# Use local Python environment for development
source .venv/bin/activate
python src/serve/app.py  # Local FastAPI development
```

### 2. Full Pipeline Testing

```bash
# Start all services
docker-compose up --build -d

# Access Airflow and trigger DAGs
# Visit http://localhost:8080

# Test model serving
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "income": 50000, "experience": 10, "education_level": "bachelor"}'
```

### 3. Debugging Services

```bash
# View specific service logs
docker-compose logs -f airflow-webserver
docker-compose logs -f mlflow
docker-compose logs -f fastapi

# Connect to service container
docker-compose exec fastapi bash
docker-compose exec mlflow bash

# Restart specific service
docker-compose restart fastapi
```

## Database Setup

The PostgreSQL service automatically creates:
- `mlflow` database with `mlflow` user
- `airflow` database with `airflow` user
- Proper permissions and roles

Database initialization is handled by `init-db.sql` script.

## Network Configuration

All services communicate via the `mlops-network` bridge network:
- Service discovery by container names (e.g., `mlflow:5000`)
- Internal communication isolated from host
- Only specified ports exposed to host system

## Production Considerations

### Security
- Change default passwords in production
- Use environment files for sensitive variables
- Enable authentication for MLflow and FastAPI
- Configure TLS/SSL certificates

### Scaling
- Replace LocalExecutor with CeleryExecutor for Airflow scaling
- Use external PostgreSQL cluster for high availability
- Add load balancer for FastAPI instances
- Configure shared storage for artifacts (S3/MinIO)

### Monitoring
- Add Prometheus metrics collection
- Configure centralized logging (ELK/Grafana)
- Set up alerting for service health
- Monitor resource usage and performance

## Troubleshooting

### Common Issues

**Service won't start:**
```bash
# Check logs for specific error
docker-compose logs [service-name]

# Verify dependencies are healthy
docker-compose ps
```

**Database connection issues:**
```bash
# Verify PostgreSQL is running
docker-compose exec postgres psql -U postgres -l

# Check service environment variables
docker-compose exec mlflow env | grep MLFLOW
```

**Model loading failures:**
```bash
# Check MLflow artifact storage
docker-compose exec mlflow ls -la /mlruns

# Verify MLflow registry
curl http://localhost:5000/api/2.0/mlflow/registered-models/list
```

### Performance Tuning

**Resource Limits:**
Add resource constraints in docker-compose.yml:
```yaml
services:
  mlflow:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

**PostgreSQL Configuration:**
Tune PostgreSQL settings for your workload:
```yaml
postgres:
  environment:
    POSTGRES_CONFIG: |
      max_connections=200
      shared_buffers=256MB
      work_mem=4MB
```

## Contributing

When modifying the Docker infrastructure:
1. Update relevant Dockerfile for service changes
2. Update docker-compose.yml for new services/volumes
3. Update this README with new configuration
4. Test the complete stack with `docker-compose up --build`
5. Verify all health checks pass