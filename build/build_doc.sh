#!/bin/bash
#
# Build and publish all docs into Pulish all notebooks to mxnet.
set -x
set -e

# prepare the env
conda env update -f build/build.yml
source activate gluon_zh_docs

make html

DSTDIR=~/zh/latest
rm -rf $DSTDIR
mv _build/html $DSTDIR

if [ -x "$(command -v xelatex)" ]; then
    make latex
    cd _build/latex
    TEXFILE=gluon_tutorials_zh.tex
    sed -i s/{tocdepth}{0}/{tocdepth}{1}/ $TEXFILE
    xelatex -interaction nonstopmode $TEXFILE
    xelatex -interaction nonstopmode $TEXFILE
    cp *.pdf $DSTDIR/
fi
