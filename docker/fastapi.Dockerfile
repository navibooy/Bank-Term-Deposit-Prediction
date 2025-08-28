FROM python:3.12-slim

WORKDIR /app

# Install uv (Astral)
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install all project deps into the image (includes FastAPI, MLflow, etc.)
# This will also install apache-airflow because it's in default deps; that's okay for now.
RUN uv sync --no-dev --frozen

# Copy source code
COPY src/ /app/src
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Start FastAPI
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]