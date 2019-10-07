FROM datadrone/deeplearn_minimal:cuda-10.1

LABEL maintainer="Brian Law <bpl.law@gmail.com>"

RUN conda install -y numpy scipy matplotlib scikit-image scikit-learn
# C++ Jupyter kernel for interfacing with opencv
RUN conda install -y xeus-cling -c conda-forge

USER root

RUN apt-get update && \
    apt-get -y install software-properties-common

RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && \
    apt-get update && \
    apt-get -y install pkg-config yasm gfortran && \
    apt-get -y install libjpeg8-dev libjasper1 libjasper-dev libpng-dev

RUN apt-get -y install libtiff5-dev libtiff-dev

RUN apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev && \
    apt-get -y install libxine2-dev libv4l-dev

WORKDIR  /usr/include/linux
RUN sudo ln -s -f ../libv4l1-videodev.h videodev.h

RUN apt-get -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev && \
    apt-get -y install libgtk-3-dev libtbb-dev qt5-default && \
    apt-get -y install libatlas-base-dev && \
    apt-get -y install libfaac-dev libmp3lame-dev libtheora-dev && \
    apt-get -y install libvorbis-dev libxvidcore-dev && \
    apt-get -y install libopencore-amrnb-dev libopencore-amrwb-dev && \
    apt-get -y install libavresample-dev && \
    apt-get -y install x264 v4l-utils libglib2.0-0

# Optional dependencies
RUN apt-get -y install libprotobuf-dev protobuf-compiler && \
    apt-get -y install libgoogle-glog-dev libgflags-dev && \
    apt-get -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

# add java which tf uses
RUN apt-get -y install openjdk-8-jdk

WORKDIR /opt

# build opencv
ENV OPENCV_VERSION="4.1.1"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& wget -O opencv_contrib.zip  https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& unzip opencv_contrib.zip \
&& mkdir /opt/opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opt/opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
  -DENABLE_PRECOMPILED_HEADERS=OFF \
  -DBUILD_opencv_java=ON \
  -DBUILD_OPENCV_PYTHON=ON \
  -DWITH_CUDA=ON \
  -DCUDA_FAST_MATH=ON \
  -DCUDA_ARCH_BIN=7.5 \
  -DOPENCV_ENABLE_NONFREE=ON \
  -DENABLE_FAST_MATH=1 \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
  -DBUILD_opencv_python2=OFF \
  -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=/opt/conda/bin/python  \
  -DPYTHON_DEFAULT_EXECUTABLE=$(which python) \
  .. \ 
&& make install -j8

RUN rm /opt/${OPENCV_VERSION}.zip

RUN conda install -y tensorflow-gpu tensorboard jupyter numpy pip matplotlib
RUN pip install boto3 flask-dropzone flask-uploads requests jsonpickle flask Pillow opencv-python

COPY . /app
WORKDIR /app

ENTRYPOINT [ "python" ]
CMD [ "main.py" ]