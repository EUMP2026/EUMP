FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/urban_embedding_v2

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build
COPY requirements.txt /tmp/build/requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/build/requirements.txt

WORKDIR /workspace/urban_embedding_v2
CMD ["bash"]
