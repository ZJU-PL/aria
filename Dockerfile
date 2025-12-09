# syntax=docker/dockerfile:1
FROM ubuntu:22.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Change apt source to mirrors.zju.edu.cn
RUN sed -i s@/archive.ubuntu.com/@/mirrors.zju.edu.cn/@g /etc/apt/sources.list
RUN apt-get clean && apt-get update

# Install dependencies
RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    vim \
    tmux \
    wget \
    curl \
    # for Yices2
    libgmp-dev\
    swig \
    cmake \
    autoconf \
    gperf \
    libboost-all-dev \
    build-essential \
    default-jre \
    zip

# Create working directory
WORKDIR /aria

# Copy project files
COPY . .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install project in development mode
RUN pip install -e .

# Download additional binary solvers
RUN python bin_solvers/download.py

# Set working directory
WORKDIR /aria
