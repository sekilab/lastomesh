FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y \
    liblas-bin \
    python3 \
    python3-pip \
    libgl1-mesa-dev \
    git

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt
