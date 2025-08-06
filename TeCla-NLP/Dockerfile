# Use multi-stage build for a smaller final image (CUDA+PyTorch)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime 

# Set environment variables in one layer. Set CUDA environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Set the working directory
WORKDIR /workspace

# Fix time synchronization issue and install system dependencies
RUN echo "Acquire::Check-Valid-Until \"false\";" > /etc/apt/apt.conf.d/99no-check-valid && \
    apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        libgl1-mesa-glx \
        python3-pip \
        nvidia-cuda-toolkit \
        texlive \
        ca-certificates \
        apt-utils && \
    apt-get clean && \
    find /var/lib/apt/lists -type f -delete

# Copy requirements file first, separately
COPY requirements.txt .

# Upgrade pip and install Python requirements
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root user with same UID/GID as host
ARG HOST_UID=1000
ARG HOST_GID=1000

RUN groupadd -g ${HOST_GID} appuser && \
    useradd -m -u ${HOST_UID} -g ${HOST_GID} appuser && \
    echo "appuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Make port 8883 available to the world outside this container
EXPOSE 8889

# Default user inside the container remains as root
USER root

# Final command
CMD ["/bin/bash"]

