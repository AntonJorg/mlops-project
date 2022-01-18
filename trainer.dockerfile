# Base image
FROM pure/python:3.7-cuda10.1-cudnn7-runtime

# privileges, maybe find way to avoid using root user
USER root

# create base directory

# set working directory to mlops folder


RUN  apt-get update \
  && apt-get install -y git \
  && rm -rf /var/lib/apt/lists/*

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# install wget

ENV PATH $PATH:/usr/local/lib/python3.7/bin
ENV PATH $PATH:/usr/local/lib/python/bin


ADD https://api.github.com/repos/AntonJorg/mlops-project/git/refs/heads/cloud_building version.json
RUN git clone -b cloud_building https://github.com/AntonJorg/mlops-project.git && \
    pip install -r requirements.txt --no-cache-dir && \
    cd mlops-project && \
    dvc pull

WORKDIR mlops-project

# install dependencies


ENV WANDB_API_KEY $WANDB_API_KEY

# run training script, -u redirects output to local console
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]