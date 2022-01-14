# Base image
FROM pure/python:3.7-cuda10.1-cudnn7-runtime

# privileges, maybe find way to avoid using root user
USER root

# create base directory

# set working directory to mlops folder


RUN  apt-get update \
  && apt-get install -y wget git \
  && rm -rf /var/lib/apt/lists/*

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# install wget

RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

ENV PATH $PATH:/root/tools/google-cloud-sdk/bin

COPY gcloud_key.json gcloud_key.json
ENV GOOGLE_APPLICATION_CREDENTIALS=gcloud_key.json
RUN gcloud auth activate-service-account --key-file=gcloud_key.json && \
    gcloud --quiet config set project dtumlops-detr

COPY requirements1.txt requirements1.txt
RUN pip install -r requirements1.txt

ENV PATH $PATH:/usr/local/lib/python3.7/bin
ENV PATH $PATH:/usr/local/lib/python/bin
RUN git clone https://github.com/AntonJorg/mlops-project.git && \
    cd mlops-project
    #&& \     dvc pull

WORKDIR mlops-project

COPY data/ data/
# install dependencies
RUN pip install -r requirements.txt --no-cache-dir

# run training script, -u redirects output to local console
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]