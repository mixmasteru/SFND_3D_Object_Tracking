FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG version=4.1.0

RUN apt-get update && apt-get upgrade -y && apt-get install -y libgl1-mesa-glx libgl1-mesa-dri && rm -rf /var/lib/apt/lists/*
RUN apt-get update \
  && apt-get install -y ssh \
      build-essential \
      gcc \
      g++ \
      gdb \
      clang \
      cmake \
      rsync \
      tar \
      python \
      git \
      libgtk2.0-dev \
      pkg-config  \
      libavcodec-dev \
      libavformat-dev \
      libswscale-dev \
      python-dev \
      python-numpy \
      libtbb2 \
      libtbb-dev \
      libjpeg-dev \
      libpng-dev \
      libtiff-dev \
      libdc1394-22-dev \
      qtbase5-dev \
      qtdeclarative5-dev \
  && apt-get clean

RUN cd ~/ && git clone --depth 1 --branch $version https://github.com/opencv/opencv.git \
&& git clone --depth 1 --branch $version https://github.com/opencv/opencv_contrib.git
RUN mkdir -p ~/opencv/build && cd ~/opencv/build \
&& cmake -D CMAKE_BUILD_TYPE=Release \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules/ \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D BUILD_opencv_legacy=OFF \
-D WITH_QT=ON \
-D WITH_GTK=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D INSTALL_C_EXAMPLES=ON .. \
&& make -j4 \
&& make install

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/my_sshd_config \
  && mkdir /run/sshd

RUN echo "root:root" | chpasswd

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/my_sshd_config"]
