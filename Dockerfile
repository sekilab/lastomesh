FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y liblas-bin python3 python3-pip

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt
