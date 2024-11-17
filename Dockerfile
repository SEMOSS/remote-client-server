ARG BASE_REGISTRY=docker.io
ARG BASE_IMAGE=nvidia/cuda
ARG BASE_TAG=12.4.0-devel-ubuntu22.04

FROM ${BASE_REGISTRY}/${BASE_IMAGE}:${BASE_TAG} AS builder

LABEL maintainer="semoss@semoss.org"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.10 \
    UV_PROJECT_ENVIRONMENT=/app \
    # Additional UV optimizations
    UV_NO_BINARY=:all: \
    UV_SYSTEM_PYTHON=1 \
    UV_WHEELTAG=py310 \
    PIP_NO_CACHE_DIR=false \
    PIP_CACHE_DIR=/root/.cache/pip

COPY requirements.txt uv.lock ./

RUN uv pip install --no-cache \
    $([ -f uv.lock ] && echo "--requirement uv.lock" || echo "--requirement requirements.txt")

# RUN if [ "$INSTALL_FLASH_ATTENTION" = "true" ]; then \
#         pip3 install packaging ninja && \
#         git clone https://github.com/HazyResearch/flash-attention.git /tmp/flash-attention && \
#         pip3 install /tmp/flash-attention --no-build-isolation; \
#     fi

COPY server server

ENV PYTHONPATH="/app/server" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HOST=0.0.0.0 \
    PORT=8888 \
    MODEL=gliner-multi-v2-1

EXPOSE ${PORT}

CMD ["sh", "-c", "python3 server/main.py --host $HOST --port $PORT --model $MODEL"]
