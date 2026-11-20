FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TFLITE_MODEL_PATH=/app/artifacts/load_mlp_float32.tflite

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" -r requirements-runtime.txt

COPY edge_features.py edge_service.py ./
COPY static ./static
COPY artifacts/load_mlp_float32.tflite ./artifacts/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "edge_service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
