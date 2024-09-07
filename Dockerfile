ARG BASE_REGISTRY=docker.io
ARG BASE_IMAGE=nvidia/cuda
ARG BASE_TAG=12.4.0-runtime-ubuntu22.04

FROM ${BASE_REGISTRY}/${BASE_IMAGE}:${BASE_TAG} as builder

LABEL maintainer="semoss@semoss.org"

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH="/app/server" 
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HOST=0.0.0.0
ENV PORT=8888
ENV TYPE=image

EXPOSE ${PORT}

RUN adduser --system --no-create-home appuser
USER appuser

CMD ["sh", "-c", "python3 server.py --host $HOST --port $PORT --type $TYPE"]
