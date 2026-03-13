# For Running NVIDIA FLARE in a Docker container, see
# https://nvflare.readthedocs.io/en/main/quickstart.html#containerized-deployment-with-docker
# This Dockerfile is primarily for building Docker images to publish for dashboard.

FROM python:3.8

RUN apt-get update && apt-get install -y wget unzip make && \
    pip install -U pip

RUN pip install nvflare

RUN wget https://github.com/nymeria-42/federated_clustering/archive/refs/heads/main.zip && \
    unzip main.zip && \
    mv federated_clustering-main/dfanalyzer/dfa-lib-python ./dfa-lib-python && \
    mv federated_clustering-main/requirements.txt ./requirements.txt && \
    mv federated_clustering-main/fed-clustering ./fed-clustering 

RUN pip install -r requirements.txt

RUN cd dfa-lib-python && make install

RUN git config --global user.email "experiment01@aws" && \
    git config --global user.name "Experiment01" && \
    git config --global init.defaultBranch main 

RUN chmod +x /fed-clustering/start_trial.sh && \
    cd /fed-clustering && \
    ./start_trial.sh

WORKDIR /fed-clustering