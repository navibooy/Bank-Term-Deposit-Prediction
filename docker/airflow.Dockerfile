FROM apache/airflow:2.7.3-python3.10

USER root
WORKDIR /opt/airflow

# Copy your code so DAGs can import src/*
COPY dags/ /opt/airflow/dags
COPY src/ /opt/airflow/src

# Make src importable
ENV PYTHONPATH="/opt/airflow:${PYTHONPATH}"

# Install ONLY the extras needed by DAG tasks (not apache-airflow)
# (versions match your pyproject to avoid surprises)
RUN pip install --no-cache-dir \
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
    shap==0.48.0 \
    pyyaml

# Optional: avoid Airflow example DAGs
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False

USER airflow