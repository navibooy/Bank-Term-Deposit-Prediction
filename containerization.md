I need you to implement the containerization component of an MLOps system. Create the Docker infrastructure for a bank marketing prediction ML system with the following components:

### Requirements to Implement

**1. Create Dockerfiles in docker/ directory:**

- `docker/airflow.Dockerfile`:
  - Base: python:3.12-slim
  - Install dependencies from pyproject.toml
  - Include Airflow webserver and scheduler components
  - Copy src/ directory and dags/
  - Set appropriate CMD for Airflow services

- `docker/mlflow.Dockerfile`:
  - Base: python:3.12-slim
  - Install MLflow with optional PostgreSQL/MinIO backend support
  - Copy src/ directory for logging integration
  - Expose port 5000
  - Set CMD for MLflow server

- `docker/fastapi.Dockerfile`:
  - Base: python:3.12-slim
  - Install FastAPI, uvicorn, and ML dependencies
  - Copy src/ directory
  - Expose port 8000
  - CMD: uvicorn with host 0.0.0.0

**2. Create docker-compose.yml in project root:**
- Define services: airflow-webserver, airflow-scheduler, mlflow, fastapi, postgres (for MLflow backend)
- Configure networking between services (FastAPI must communicate with MLflow)
- Set environment variables: MLFLOW_TRACKING_URI, database credentials
- Define volumes for data persistence: ./mlruns:/mlruns, ./data:/app/data
- Map ports: 8080 (Airflow), 5000 (MLflow), 8000 (FastAPI)
- Use build contexts pointing to respective Dockerfiles

**3. Configuration Requirements:**
- Use environment variables for service communication
- Ensure MLflow tracking URI is accessible from all services
- Configure PostgreSQL as MLflow backend database
- Set up proper volume mounts for data directories and MLflow artifacts

**4. Service Dependencies:**
- Airflow depends on MLflow for experiment logging
- FastAPI depends on MLflow for model loading
- MLflow depends on PostgreSQL for metadata storage

**Key Constraints:**
- All services must use the same Python 3.12-slim base image
- Dependencies must be installed from pyproject.toml/requirements.txt
- Services must be able to communicate via service names in docker-compose network
- Data persistence through volumes is critical
- Follow the exact port mappings specified

Create all necessary files with proper error handling, health checks where appropriate, and ensure the entire system can be started with `docker-compose up --build`. Include comments explaining the networking and dependency setup.