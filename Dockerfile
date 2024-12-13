ARG BASE_REGISTRY=docker.io
ARG BASE_IMAGE=nvidia/cuda
ARG BASE_TAG=12.4.0-devel-ubuntu22.04

FROM ${BASE_REGISTRY}/${BASE_IMAGE}:${BASE_TAG} AS builder

LABEL maintainer="semoss@semoss.org"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV HOME=/root

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    curl \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /root/.config/git

RUN git config --system credential.helper store \
    && git config --system http.sslVerify false \
    && git config --system http.postBuffer 524288000 \
    && git config --system http.lowSpeedLimit 1000 \
    && git config --system http.lowSpeedTime 600 \
    && git config --system protocol.version 2 \
    && git config --system lfs.concurrenttransfers 4 \
    && git config --system core.compression 0 \
    && git config --system http.maxRequestBuffer 100M

WORKDIR /app

RUN mkdir -p /app/model_files/.cache \
    && chmod -R 777 /app/model_files \
    && mkdir -p /root/.cache \
    && chmod -R 777 /root/.cache

ENV HOME=/root \
    GIT_LFS_SKIP_SMUDGE=0 \
    GIT_TERMINAL_PROMPT=0 \
    GIT_LFS_PROGRESS=true \
    GIT_TRACE=1 \
    TRANSFORMERS_CACHE=/app/model_files/.cache \
    HF_HOME=/app/model_files/.cache \
    HUGGINGFACE_HUB_CACHE=/app/model_files/.cache \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.10 \
    UV_PROJECT_ENVIRONMENT=/app \
    UV_NO_BINARY=:all: \
    UV_SYSTEM_PYTHON=1 \
    UV_WHEELTAG=py310 \
    PIP_NO_CACHE_DIR=false \
    PIP_CACHE_DIR=/root/.cache/pip \
    PYTHONPATH="/app/server" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HOST=0.0.0.0 \
    PORT=8888 \
    MODEL=gliner-multi-v2-1

COPY requirements.txt uv.lock pyproject.toml ./

RUN uv pip install -r pyproject.toml

COPY server server

RUN git lfs install --system --skip-repo \
    && chown -R root:root /app/model_files \
    && chmod -R 777 /app/model_files \
    && chown -R root:root /root/.cache \
    && chmod -R 777 /root/.cache

EXPOSE ${PORT}

CMD ["sh", "-c", "python3 server/main.py --host $HOST --port $PORT --model $MODEL"]