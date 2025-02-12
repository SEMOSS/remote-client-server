ARG BASE_IMAGE
FROM ${BASE_IMAGE}
# FROM docker.semoss.org/genai/remote-client-server-base:latest
# FROM remote-client-server-base

LABEL maintainer="semoss@semoss.org"

WORKDIR /app

RUN mkdir -p /app/model_files/.cache \
    && chmod -R 777 /app/model_files \
    && mkdir -p /root/.cache \
    && chmod -R 777 /root/.cache

ENV TRANSFORMERS_CACHE=/app/model_files/.cache \
    HF_HOME=/app/model_files/.cache \
    HUGGINGFACE_HUB_CACHE=/app/model_files/.cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    UV_PROJECT_ENVIRONMENT=/app \
    PYTHONPATH="/app/server" \
    HOST=0.0.0.0 \
    PORT=8888

COPY uv.lock pyproject.toml ./

RUN uv pip install -r pyproject.toml

COPY server server

EXPOSE ${PORT}

CMD ["sh", "-c", "python3 server/main.py --host $HOST --port $PORT --model $MODEL"]