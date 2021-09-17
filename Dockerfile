FROM ubuntu:18.04
# 
RUN set -x
# 
# # Don't download stuff to the git repo, that's messy.
# 
# Update packages
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install bzip2

RUN apt-get install -y wget

ARG ANACONDA_INSTALLER="Anaconda3-2020.11-Linux-x86_64.sh"
RUN wget "https://repo.continuum.io/archive/$ANACONDA_INSTALLER"
RUN chmod +x $ANACONDA_INSTALLER
RUN ./$ANACONDA_INSTALLER -b

RUN /bin/bash -c "source ${HOME}/.bashrc"

RUN ${HOME}/anaconda3/bin/pip install --upgrade pip
RUN ${HOME}/anaconda3/bin/pip install --upgrade tensorflow==2.4.1
RUN ${HOME}/anaconda3/bin/jupyter notebook --generate-config

RUN apt-get install -y python3-pip
RUN pip3 install transformers
RUN /bin/bash -c "echo export PATH=~/anaconda3/bin:$PATH >> ${HOME}/.bashrc"
RUN apt-get install -y vim
RUN apt-get update
RUN apt-get install git -y
RUN pip3 install pydicom
RUN git clone https://github.com/pieper/dicomsort.git
RUN pip3 install pylidc
