FROM apache/airflow:2.7.3-python3.10

USER root
WORKDIR /opt/airflow

# Copy your code so DAGs can import src/*
COPY dags/ /opt/airflow/dags
COPY src/ /opt/airflow/src

# Make src importable
ENV PYTHONPATH="/opt/airflow:${PYTHONPATH}"

# Switch to airflow user BEFORE installing extras
USER airflow

# Upgrade pip + core tools first (reduces build failures)
RUN pip install --upgrade pip setuptools wheel

# Install only the dependencies you need for your DAGs
# Add --timeout to avoid ReadTimeout errors on slow connections
RUN pip install --user --no-cache-dir --timeout=120 \
    catboost==1.2.8 \
    evidently==0.7.11 \
    fastapi==0.116.1 \
    kaggle==1.7.4.5 \
    matplotlib==3.10.5 \
    mlflow==3.3.1 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.2 \
    seaborn==0.13.2 \
    shap==0.45.1 \
    pyyaml

# Optional: avoid Airflow example DAGs
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False