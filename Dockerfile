FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         python3 \
         python3-pip \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Install face-alignment package
WORKDIR /workspace
RUN chmod -R a+w /workspace
RUN git clone https://github.com/1adrianb/face-alignment
WORKDIR /workspace/face-alignment
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .
