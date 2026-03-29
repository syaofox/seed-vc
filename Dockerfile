FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PYTHON=3.10
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_HTTP_TIMEOUT=300
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:$PATH"

WORKDIR /app

COPY . .

RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

RUN --mount=type=cache,target=/tmp/uv-cache,uid=1000,gid=1000 \
    UV_CACHE_DIR=/tmp/uv-cache uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

EXPOSE 7860 7861

ENTRYPOINT ["docker/entrypoint.sh"]
