FROM nvidia/cuda:10.2-base-ubuntu18.04

# Install newest CUDA: https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130

RUN apt-get update && \
    apt-get upgrade -y && \
    apt install -y git && \
    apt-get install -y cmake && \
    apt-get install -y xorg-dev && \
    apt-get install -y libboost-all-dev && \
    apt-get install -y libglew-dev && \
    apt-get install -y libcgal-dev && \
    apt-get install -y libtbb-dev

# Install/Configure VirtualGL: https://virtualgl.org/vgldoc/2_2_1/#hd004001
#   4.1 - Installing VirtualGL on Linux
#   5.1 - Configure as a VirtualGL Server


WORKDIR /usr/viper
COPY . /usr/viper
# Let it fail so we can grab it's dependencies with apt --fix-broken install 
RUN dpkg -i /usr/viper/virtualgl_2.6.3_amd64.deb; exit 0
RUN apt --fix-broken install -y
RUN apt-get upgrade -y
RUN dpkg -i /usr/viper/virtualgl_2.6.3_amd64.deb
RUN /opt/VirtualGL/bin/vglserver_config -config +s +f -t

# TODO:
# Follow VIPER GitHub instructions: https://github.com/vcg-uvic/viper
# And here are the bash commands I use to run the demo and write the output to out.mpg (
