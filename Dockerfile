FROM ros:humble-ros-base-jammy AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y --no-install-recommends \
    xserver-xorg

RUN mkdir /workspace
WORKDIR /workspace
COPY . /workspace/

FROM base AS dev

RUN apt update && apt install -y --no-install-recommends \
    vim \
    tmux \
    x11-apps

CMD ["/bin/bash"]

