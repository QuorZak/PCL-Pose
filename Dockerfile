# Build environment for PCL-Pose on Debian 13 (trixie), whose packages include OpenCV 4.10 (4.7 or newer is needed)
# and PCL 1.15. Debian has no librealsense package, so it is built from source with the RSUSB backend, which needs no
# kernel modules and is enough to compile and link the programs.
#
#   docker build -t pcl-pose .                      # builds the dependencies, then compiles all the programs
#   docker build --target deps -t pcl-pose-deps .   # only the dependencies, e.g. for a development shell:
#   docker run --rm -it -v "$PWD":/src -w /src pcl-pose-deps bash
#
# This image only compiles the code. Running the programs needs a RealSense camera on the host, which is not set up here.

FROM debian:trixie AS deps

ARG LIBREALSENSE_VERSION=v2.55.1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config ca-certificates \
        libpcl-dev libopencv-dev libflann-dev libhdf5-dev libboost-all-dev libeigen3-dev \
        libusb-1.0-0-dev libssl-dev libudev-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${LIBREALSENSE_VERSION} https://github.com/IntelRealSense/librealsense.git /tmp/librealsense \
    && cmake -S /tmp/librealsense -B /tmp/librealsense/build \
        -DCMAKE_BUILD_TYPE=Release -DFORCE_RSUSB_BACKEND=ON \
        -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    && cmake --build /tmp/librealsense/build --parallel "$(nproc)" \
    && cmake --install /tmp/librealsense/build \
    && ldconfig \
    && rm -rf /tmp/librealsense

FROM deps AS build

WORKDIR /src
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src
COPY apps ./apps
COPY config ./config

# The default programs, then the experimental pose program that is left out of the default build
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel "$(nproc)" \
    && cmake --build build --parallel "$(nproc)" --target pose
