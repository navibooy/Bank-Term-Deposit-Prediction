# MLflow Dockerfile for Bank Term Deposit Prediction MLOps Pipeline
FROM python:3.12-slim

# Set environment variables
ENV MLFLOW_HOME=/opt/mlflow
ENV MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlruns

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create mlflow user and directories
RUN groupadd -r mlflow && useradd -r -g mlflow mlflow
RUN mkdir -p $MLFLOW_HOME /mlruns /mlartifacts
RUN chown -R mlflow:mlflow $MLFLOW_HOME /mlruns /mlartifacts

# Set working directory
WORKDIR /app

# Install essential dependencies for MLflow
RUN pip install --no-cache-dir \
    mlflow==3.3.1 \
    psycopg2-binary \
    boto3 \
    pandas==2.3.2 \
    numpy==2.2.6 \
    scikit-learn==1.7.1 \
    pyyaml

# Copy source code for logging integration
COPY src/ ./src/
COPY config.yaml ./

# Set proper ownership
RUN chown -R mlflow:mlflow /app

# Switch to mlflow user
USER mlflow

# Expose MLflow server port
EXPOSE 5000

# Health check for MLflow server
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Default command to start MLflow server
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow", \
     "--default-artifact-root", "/mlruns", \
     "--artifacts-destination", "/mlruns"]