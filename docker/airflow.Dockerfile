# Airflow Dockerfile for Bank Term Deposit Prediction MLOps Pipeline
FROM python:3.12-slim

# Set environment variables for Airflow
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__FERNET_KEY=46BKJoQYlPPOexq0OhDZnIlNepKFf87WFwLbfzqDDho=
ENV AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true
ENV AIRFLOW__CORE__LOAD_EXAMPLES=false
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=true
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow

# Set matplotlib environment variables to avoid permission issues
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV MPLBACKEND=Agg

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create airflow user and directories
RUN groupadd -r airflow && useradd -r -g airflow airflow
RUN mkdir -p $AIRFLOW_HOME/dags $AIRFLOW_HOME/logs $AIRFLOW_HOME/plugins
RUN chown -R airflow:airflow $AIRFLOW_HOME

# Set working directory
WORKDIR /app

# Install dependencies - Airflow core + ML pipeline dependencies
RUN pip install --no-cache-dir \
    apache-airflow[postgres]==3.0.4 \
    flask-appbuilder \
    psycopg2-binary \
    asyncpg \
    pyyaml \
    # ML Pipeline Dependencies \
    catboost==1.2.8 \
    evidently==0.7.11 \
    fastapi==0.116.1 \
    kaggle==1.7.4.5 \
    matplotlib==3.10.5 \
    mlflow==3.3.1 \
    numpy==2.2.6 \
    pandas==2.3.2 \
    scikit-learn==1.7.1 \
    seaborn==0.13.2 \
    shap==0.48.0

# Copy source code and DAGs
COPY src/ ./src/
COPY dags/ $AIRFLOW_HOME/dags/
COPY config.yaml ./

# Set proper ownership
RUN chown -R airflow:airflow /app
RUN chown -R airflow:airflow $AIRFLOW_HOME

# Switch to airflow user
USER airflow

# Expose Airflow webserver port
EXPOSE 8080

# Health check for Airflow webserver
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Default command (can be overridden by docker-compose)
CMD ["airflow", "webserver"]