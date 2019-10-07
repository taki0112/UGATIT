FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN pip3 install boto3 flask-dropzone flask-uploads requests jsonpickle flask Pillow opencv-python matplotlib numpy

RUN apt-get update && apt-get install -y curl lsb-release sudo rsync libsm6 libxext6 libxrender-dev
RUN export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` \
    && echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update && apt-get install -y gcsfuse

RUN mkdir -p /app/checkpoint

COPY . /app
WORKDIR /app

ENTRYPOINT [ "python3" ]
CMD [ "main.py" ]