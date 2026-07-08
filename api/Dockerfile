FROM python:3.11-slim
WORKDIR /app
# libgomp1: python:3.11-slim strips GCC runtime libs, but lightgbm's compiled
# extension (and torch/faiss) link against libgomp.so.1 at import/load time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/ ./api/
COPY src/ ./src/
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
