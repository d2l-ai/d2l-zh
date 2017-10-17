#!/bin/bash
#
# Build and publish all docs into Pulish all notebooks to mxnet.
set -x
set -e

# prepare the env
conda env update -f build/build.yml
source activate gluon_zh_docs

make html

TEX=gluon_tutorials_zh.tex
PDF=gluon_tutorials_zh.pdf
DSTDIR=~/zh/latest
if [ -e $DSTDIR/$PDF ]; then
    cp $DSTDIR/$PDF build/_build/html/
fi
rm -rf $DSTDIR

rm -rf build/data
make pkg

mv build/_build/html $DSTDIR

set +e
if [ -x "$(command -v xelatex)" ]; then
    make latex
    cd build/_build/latex
    sed -i s/{tocdepth}{0}/{tocdepth}{1}/ $TEX
    xelatex -interaction nonstopmode $TEX
    xelatex -interaction nonstopmode $TEX
    cp $PDF $DSTDIR/
fi

aws s3 sync --delete $DSTDIR/ s3://zh.gluon.ai/ --acl public-read
