# ══════════════════════════════════════════════════════════════
# Stage 1: BUILDER - instala dependencias en wheels pre-compilados
# ══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libpq-dev libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel

# Dependencias CPU-only para PyTorch
RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    fastapi "uvicorn[standard]" gunicorn python-multipart \
    pillow opencv-python-headless numpy scikit-learn \
    pyyaml

# ══════════════════════════════════════════════════════════════
# Stage 2: RUNTIME - imagen ligera sin compiladores
# ══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 libffi8 curl libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && \
    rm -rf /wheels

COPY src/ ./src/
COPY config/ ./config/
RUN pip install --no-cache-dir --no-deps -e . || pip install --no-cache-dir --no-deps .

RUN mkdir -p checkpoints models logs && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["gunicorn", "facial_attributes.api.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
