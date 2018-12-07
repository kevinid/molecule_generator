FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

LABEL maintainer="kevinid4g@gmail.com"

# ========== Install required packages ==========

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        cmake \
        git \
        wget \
        curl \
        unzip \
        openssh-server \
        tmux \
        libffi-dev \
        python2.7 \
        python2.7-dev \
        python-pip \
        tmux \
        less \
        zip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip --no-cache-dir install --upgrade pip && \
    pip2.7 --no-cache-dir install setuptools && \
    pip2.7 --no-cache-dir install \
        numpy \
        pandas \
        scipy \
        sklearn \
        jupyter \
        matplotlib \
        lmdb \
        PyMySQL \
        networkx \
        protobuf \
        mxnet-cu90==1.3.1b20181004

# ========== Build RDKit from source ==========
# Install required dependences
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        libboost-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libboost-serialization-dev \
        libboost-python-dev \
        libboost-regex-dev \
        libcairo2-dev \
        libeigen3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download from github
ARG RDKIT_VERSION=Release_2018_03_3
RUN wget --quiet https://github.com/rdkit/rdkit/archive/${RDKIT_VERSION}.tar.gz && \
    tar -xzf ${RDKIT_VERSION}.tar.gz && \
    mv rdkit-${RDKIT_VERSION} rdkit && \
    rm ${RDKIT_VERSION}.tar.gz

# Configure environment variables
ENV RDBASE=$PWD/rdkit
ENV LD_LIBRARY_PATH=${RDBASE}/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${RDBASE}:${PYTHONPATH}

# Install
RUN mkdir rdkit/build && cd rdkit/build && \
    cmake -D RDK_BUILD_PYTHON_WRAPPERS=ON \
    -D PYTHON_LIBRARY=/usr/lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so \
    -D PYTHON_INCLUDE_DIR=/usr/include/python2.7/ \
    -D PYTHON_EXECUTABLE=/usr/bin/python2.7 \
    .. && \
    make -j $(nproc) && \
    # make && \
    make install

# ========== Configure jupyter notebook ==========
RUN mkdir /notebooks && chmod a+rwx /notebooks
RUN mkdir /.local && chmod a+rwx /.local
EXPOSE 8888

# ========== Configure entrypoint ==========
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]