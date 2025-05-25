# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install uv
RUN uv pip install comfy-cli --system
RUN /usr/bin/yes | comfy --workspace /comfyui install --version 0.3.30 --cuda-version 12.6 --nvidia

WORKDIR /comfyui
ADD src/extra_model_paths.yaml ./

WORKDIR /
RUN uv pip install runpod requests websocket-client --system

ADD src/start.sh handler.py test_input.json ./
RUN chmod +x /start.sh

CMD ["/start.sh"]

# Stage 2: Download only flux1-dev-fp8 model
FROM base AS downloader

WORKDIR /comfyui
RUN mkdir -p models/checkpoints

RUN wget -q -O models/checkpoints/flux1-dev-fp8.safetensors \
    https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors

# Stage 3: Final image
FROM base AS final
COPY --from=downloader /comfyui/models /comfyui/models
