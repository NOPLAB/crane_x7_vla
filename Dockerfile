FROM osrf/ros:humble-desktop-full AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt install -y --no-install-recommends \
    xserver-xorg

# Install dependencies
RUN mkdir -p /tmp/ros2_dependencies/src
COPY ros2/src /tmp/ros2_dependencies/src
RUN cd /tmp/ros2_dependencies && rosdep install -r -y -i --from-paths .  && rm -rf /tmp/ros2_dependencies

RUN mkdir /workspace
WORKDIR /workspace

FROM base AS dev

RUN apt-get update && apt install -y --no-install-recommends \
    vim \
    tmux \
    iproute2 \
    x11-apps

CMD ["/bin/bash"]

