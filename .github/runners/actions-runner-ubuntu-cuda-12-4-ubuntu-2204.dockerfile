# Use NVIDIA CUDA 12.4.0 development image with Ubuntu 22.04 as the base
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Docker and Docker Compose arguments

# Use 1001 and 121 for compatibility with GitHub-hosted runners
ARG RUNNER_UID=1000
ARG DOCKER_GID=1001

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update -y \
    && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:git-core/ppa \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    dnsutils \
    ftp \
    git \
    uuid-dev \
    iproute2 \
    iputils-ping \
    jq \
    libunwind8 \
    locales \
    netcat \
    openssh-client \
    parallel \
    python3-pip \
    rsync \
    shellcheck \
    sudo \
    telnet \
    time \
    tzdata \
    unzip \
    upx \
    wget \
    lsb-release \
    openssl \
    libssl-dev \
    manpages-dev \
    zip \
    zstd \
    pkg-config \
    ccache \
    gcc \
    g++ \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Add Kitware's APT repository for CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get update && \
    apt-get install -y cmake

# Download latest git-lfs version
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y --no-install-recommends git-lfs

ARG RUNNER_VERSION=2.319.1

RUN adduser --disabled-password --gecos "" --uid $RUNNER_UID runner \
&& groupadd docker --gid $DOCKER_GID \
&& usermod -aG sudo runner \
&& usermod -aG docker runner \
&& echo "%sudo   ALL=(ALL:ALL) NOPASSWD:ALL" > /etc/sudoers \
&& echo "Defaults env_keep += \"DEBIAN_FRONTEND\"" >> /etc/sudoers

ENV HOME=/home/runner

# cd into the user directory, download and unzip the github actions runner
RUN cd /home/runner && mkdir actions-runner && cd actions-runner \
    && curl -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    && tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

RUN chown -R runner:runner /home/runner && /home/runner/actions-runner/bin/installdependencies.sh

ADD ./start.sh /home/runner/start.sh

RUN chmod +x /home/runner/start.sh

# Update the CUDA compatibility libraries path for 12.4
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.4/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

ENTRYPOINT ["/bin/bash", "/home/runner/start.sh"]

USER runner