ARG BASE_REGISTRY=docker.io
ARG BASE_IMAGE=diffusers/diffusers-pytorch-xformers-cuda
ARG BASE_TAG=latest

FROM ${BASE_REGISTRY}/${BASE_IMAGE}:${BASE_TAG} as builder


LABEL maintainer="semoss@semoss.org"

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/model_files/pixart && chmod 777 /app/model_files/pixart


ENV PYTHONPATH="/app/server" 
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HOST=0.0.0.0
ENV PORT=8888
ENV MODEL=pixart



EXPOSE ${PORT}

CMD ["sh", "-c", "python3 server/main.py --host $HOST --port $PORT"]
