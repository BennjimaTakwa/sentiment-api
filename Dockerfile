FROM python:3.11-slim

# ── Metadata
LABEL maintainer="your.email@example.com"
LABEL description="Customer Sentiment & Retention API"
LABEL version="1.0.0"

# ── System deps (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory
WORKDIR /api

# ── Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application code
COPY app/ ./app/

# ── Non-root user for security
RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /api
USER apiuser

# ── Expose port
EXPOSE 8000

# ── Docker health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health/ready || exit 1

# ── Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]