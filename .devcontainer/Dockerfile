FROM nvidia/cuda:11.3.1-base-ubuntu20.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

EXPOSE 8000
EXPOSE 8080

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV CURL_CA_BUNDLE ''
#ENV HF_HOME "/workspace/data/huggingface_data/"

# Setup user
RUN groupadd --gid $USER_GID $USERNAME &&\
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME &&\
    apt update && \
    apt -y upgrade && \
    apt install sudo -y &&\
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME &&\
    chmod 0440 /etc/sudoers.d/$USERNAME

RUN DEBIAN_FRONTEND=noninteractive apt install software-properties-common curl ffmpeg libsm6 libxext6 -y &&\ 
    add-apt-repository ppa:deadsnakes/ppa -y &&\
    apt install -y python3.8 python3.8-distutils &&\
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8 &&\
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN apt install git -y &&\
    git config --global http.sslVerify false &&\
    # pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html &&\
    pip install protobuf


# Install pip requirements
COPY . /tmpdir
RUN cd /tmpdir && pip install -r requirements.txt && cd .. && rm -rf /tmpdir
USER ${USERNAME}

ENTRYPOINT ["tail", "-f", "/dev/null"]