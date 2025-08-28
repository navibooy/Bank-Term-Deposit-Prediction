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

# Install only the dependencies you need for your DAGs
RUN pip install --user --no-cache-dir \
    catboost==1.2.8 \
    evidently==0.7.11 \
    fastapi==0.116.1 \
    kaggle==1.7.4.5 \
    matplotlib==3.10.5 \
    mlflow==3.3.1 \
    numpy==1.26.4 \        # 👈 downgrade to 1.26.x
    pandas==2.2.2 \        # 👈 match numpy compat (2.3.2 expects NumPy 2.x)
    scikit-learn==1.5.2 \  # 👈 latest stable w/ NumPy 1.26.x
    seaborn==0.13.2 \
    shap==0.45.1 \         # 👈 also stick to <0.46 for numpy 1.x
    pyyaml

# Optional: avoid Airflow example DAGs
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False