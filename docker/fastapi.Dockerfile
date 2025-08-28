# FastAPI Dockerfile for Bank Term Deposit Prediction MLOps Pipeline
FROM python:3.12-slim

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create fastapi user and directories
RUN groupadd -r fastapi && useradd -r -g fastapi fastapi
RUN mkdir -p /app/models /app/data

# Set working directory
WORKDIR /app

# Install essential dependencies for FastAPI
RUN pip install --no-cache-dir \
    fastapi==0.116.1 \
    uvicorn[standard] \
    mlflow==3.3.1 \
    pandas==2.3.2 \
    numpy==2.2.6 \
    scikit-learn==1.7.1 \
    catboost==1.2.8 \
    pyyaml

# Copy source code
COPY src/ ./src/
COPY config.yaml ./

# Set proper ownership
RUN chown -R fastapi:fastapi /app

# Switch to fastapi user
USER fastapi

# Expose FastAPI port
EXPOSE 8000

# Health check for FastAPI server
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command to start FastAPI server with uvicorn
CMD ["uvicorn", "src.serve.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]