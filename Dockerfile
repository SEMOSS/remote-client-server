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

ENV PYTHONPATH="/app" 

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV TCP_HOST=0.0.0.0
ENV TCP_PORT=8888

ENV HTTP_HOST=0.0.0.0
ENV HTTP_PORT=8080

ENV TYPE=image

EXPOSE ${TCP_PORT}
EXPOSE ${HTTP_PORT}

CMD ["sh", "-c", "python3 tcp_server/server.py --host $TCP_HOST --port $TCP_PORT & python3 http_server/main.py"]
