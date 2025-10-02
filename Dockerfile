FROM ros:humble-ros-base-jammy AS base

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
    x11-apps

CMD [ "/workspace/scripts/docker/entrypoint.sh" ]

FROM base AS ros2_demo_real

CMD ["/workspace/ros2/scripts/demo/launch_real.sh"]

FROM base AS ros2_demo_sim

CMD ["/workspace/ros2/scripts/demo/launch_sim.sh"]

