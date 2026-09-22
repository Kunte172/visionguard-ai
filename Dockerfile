FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VISIONGUARD_MODEL_PATH=/app/artifacts/detector.joblib

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data
COPY artifacts ./artifacts
RUN visionguard generate-sample --output data/sample --seed 42 && \
    visionguard train --manifest data/sample/manifest.csv

EXPOSE 8000
CMD ["uvicorn", "visionguard_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]
