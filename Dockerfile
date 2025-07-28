# Use a more recent Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    git \
    g++ \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip

# Install PyTorch with CUDA support from PyTorch's official repository
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Copy requirements file
COPY requirements.txt .

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install detectron2 from source (more reliable than wheels)
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Copy rest of the files
COPY . .

# Copy model files
RUN mkdir -p /app/PubLayNet_model
WORKDIR /app/PubLayNet_model
RUN curl -L -o model_final.pth https://github.com/Layout-Parser/layout-parser/releases/download/v0.3.4/model_final.pth && \
    curl -L -o config.yml https://github.com/Layout-Parser/layout-parser/releases/download/v0.3.4/config.yaml

COPY PubLayNet_model /app/PubLayNet_model/
RUN pip install pyyaml  # Ensure YAML support

# Create required directories
RUN mkdir -p /app/{iopath_cache,output,input}

WORKDIR /app

# Default command
CMD ["python", "main.py"]