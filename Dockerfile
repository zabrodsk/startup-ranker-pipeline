FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
COPY web/ ./web/
COPY security/ ./security/
COPY scripts/verify_protected_scoring.py ./scripts/verify_protected_scoring.py

RUN python scripts/verify_protected_scoring.py

RUN python -m pip install --no-cache-dir --upgrade "pip>=26.1.2" && \
    pip install --no-cache-dir -e "." && \
    pip install --no-cache-dir fastapi uvicorn python-multipart

EXPOSE 8000

# Railway injects PORT; app reads it via os.getenv("PORT", "8000")
ENV PORT=8000

CMD ["python", "-m", "agent.railway_service"]
