# Base image
FROM anibali/pytorch:1.10.0-cuda11.3-ubuntu20.04

# privileges, maybe find way to avoid using root user
USER root

# create base directory
RUN mkdir "mlops"
# set working directory to mlops folder
WORKDIR "/mlops"

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy application over
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

# create model save dir
RUN mkdir "models"

# install dependencies
RUN pip install -r requirements.txt --no-cache-dir

# run training script, -u redirects output to local console
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]