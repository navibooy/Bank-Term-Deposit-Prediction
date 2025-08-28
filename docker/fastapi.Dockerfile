FROM python:3.12-slim

# Create a non-root user (id 1000 to match host dev usually)
RUN useradd -m -u 1000 fastapiuser

WORKDIR /app

# Install uv (dependency manager)
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into .venv
RUN uv sync --no-dev --frozen

# Add .venv/bin to PATH so uvicorn & deps are visible
ENV PATH="/app/.venv/bin:$PATH"

# Make sure app files belong to non-root user
RUN chown -R fastapiuser:fastapiuser /app

# Switch to non-root
USER fastapiuser

# Set Python path (source code will come from volume mount, not COPY)
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose FastAPI
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]