# docker/mlflow.Dockerfile
FROM python:3.12-slim

# (Optional) Install curl for healthchecks/debug
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and a persistent artifact dir
RUN useradd -m -u 1000 mlflow && mkdir -p /mlruns && chown -R mlflow:mlflow /mlruns

# Install MLflow (match your pyproject), plus optional Postgres driver
RUN pip install --no-cache-dir \
    mlflow==3.3.1 \
    psycopg2-binary==2.9.9

# If you plan to use S3/MinIO later, also add:
# RUN pip install --no-cache-dir boto3

WORKDIR /app
USER mlflow

# MLflow UI port
EXPOSE 5000

# We don't hardcode CMD here because docker-compose provides the server command:
#   mlflow server --host 0.0.0.0 --port 5000 \
#                 --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
#                 --default-artifact-root ${MLFLOW_ARTIFACT_ROOT}