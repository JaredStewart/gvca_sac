FROM python:3.14-slim

WORKDIR /app

# Optionally inject corporate CA cert for builds behind TLS-intercepting proxies
RUN --mount=type=secret,id=ca_cert,target=/tmp/ca-cert.pem \
    if [ -f /tmp/ca-cert.pem ]; then \
      cp /tmp/ca-cert.pem /usr/local/share/ca-certificates/custom-ca.crt && \
      update-ca-certificates; \
    fi

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Use system TLS certs so uv respects injected CA certs
ENV UV_NATIVE_TLS=1

# Copy dependency files
COPY pyproject.toml .

# Install dependencies (creates lock file if missing)
RUN uv sync --no-dev

# Copy application code
COPY app/ ./app/

# Create artifacts directory
RUN mkdir -p /app/artifacts

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
