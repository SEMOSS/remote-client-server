ARG BASE_REGISTRY=docker.io
ARG BASE_IMAGE=nvidia/cuda
ARG BASE_TAG=12.1.0-devel-ubuntu22.04

FROM ${BASE_REGISTRY}/${BASE_IMAGE}:${BASE_TAG} AS builder

LABEL maintainer="semoss@semoss.org"

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# RUN pip3 install -r requirements.txt

RUN if [ "$INSTALL_FLASH_ATTENTION" = "true" ]; then \
        pip3 install packaging ninja && \
        git clone https://github.com/HazyResearch/flash-attention.git /tmp/flash-attention && \
        pip3 install /tmp/flash-attention --no-build-isolation; \
    fi

COPY . .

RUN mkdir -p /app/model_files/pixart && chmod 777 /app/model_files/pixart
RUN mkdir -p /app/model_files/phi-3-mini-128k-instruct && chmod 777 /app/model_files/phi-3-mini-128k-instruct

ENV PYTHONPATH="/app/server" 
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HOST=0.0.0.0
ENV PORT=8888
ENV MODEL=pixart

EXPOSE ${PORT}

CMD ["sh", "-c", "python3 server/main.py --host $HOST --port $PORT --model $MODEL"]
