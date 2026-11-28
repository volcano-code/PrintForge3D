FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/PrintForge3D
COPY . .
RUN python -m pip install --upgrade pip && \
    python -m pip install -e ".[dev]"

CMD ["bash"]
