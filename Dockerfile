FROM nvidia/cuda:10.2-base-ubuntu18.04

# Follow VIPER GitHub instructions: https://github.com/vcg-uvic/viper
RUN apt-get update && \
    apt-get upgrade -y && \
    apt install -y git&& \
    apt-get install -y cmake && \
    apt-get install -y xorg-dev && \
    apt-get install -y libboost-all-dev && \
    apt-get install -y libglew-dev && \
    apt-get install -y libcgal-dev && \
    apt-get install -y libtbb-dev && \
    apt-get install -y python3-pip && \
    pip3 install torch torchvision tqdm

# Install/Configure VirtualGL: https://virtualgl.org/vgldoc/2_2_1/#hd004001
#   4.1 - Installing VirtualGL on Linux
#   5.1 - Configure as a VirtualGL Server


# Couldn't get VirtualGL to work inside of docker
COPY ./virtualgl_2.6.3_amd64.deb /tmp
# Let it fail so we can grab it's dependencies with apt --fix-broken install 
RUN dpkg -i /tmp/virtualgl_2.6.3_amd64.deb; exit 0
RUN apt --fix-broken install -y
RUN apt-get upgrade -y
RUN dpkg -i /tmp/virtualgl_2.6.3_amd64.deb
RUN /opt/VirtualGL/bin/vglserver_config -config +s +f -t

