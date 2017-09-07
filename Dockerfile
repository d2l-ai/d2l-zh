FROM ubuntu:16.04

RUN apt-get update && apt-get install -y python3 git wget bzip2

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH /root/miniconda3/bin:$PATH

RUN git clone https://github.com/mli/gluon-tutorials-zh && \
    cd gluon-tutorials-zh && \
    conda env create -f environment.yml

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN source activate gluon && \
    mkdir notebook && \
    jupyter notebook --allow-root --generate-config && \
    echo "c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'" >>~/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

CMD source activate gluon && cd /gluon-tutorials-zh && \
    jupyter notebook --ip=0.0.0.0 --allow-root
