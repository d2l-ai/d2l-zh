FROM ubuntu:16.04

# install python and conda
RUN apt-get update && apt-get install -y python3 git wget bzip2 
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     bash Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH /root/miniconda3/bin:$PATH

# install deps
COPY environment.yml /
RUN conda env create -f environment.yml

# source activate need bash
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# setup notedown
RUN source activate gluon && \
    pip install https://github.com/mli/notedown/tarball/master && \
    mkdir notebook && \
    jupyter notebook --allow-root --generate-config && \
    echo "c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'" >>~/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

# copy notebooks
RUN  mkdir /gluon-tutorials-zh
COPY / /gluon-tutorials-zh/

# sanity check
# RUN source activate gluon && notedown --run /gluon-tutorials-zh/chapter_crashcourse/ndarray.md

# for chinese supports
ENV LANG C.UTF-8

CMD source activate gluon && cd /gluon-tutorials-zh && \
    jupyter notebook --ip=0.0.0.0 --allow-root
