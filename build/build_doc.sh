#!/bin/bash
#
# Build and publish all docs into Pulish all notebooks to mxnet.
set -x
set -e

# prepare the env
conda env update -f build/build.yml
source activate gluon_zh_docs

make html

rm -rf build/data
make pkg

set +e
if [ -x "$(command -v xelatex)" ]; then
    make latex
    cd build/_build/latex
    TEX=gluon_tutorials_zh.tex
    sed -i s/{tocdepth}{0}/{tocdepth}{1}/ $TEX
    xelatex -interaction nonstopmode $TEX
    xelatex -interaction nonstopmode $TEX
    cd ../../../    
    cp build/_build/latex/gluon_tutorials_zh.pdf build/_build/html/
fi

aws s3 sync --delete build/_build/html/ s3://zh.gluon.ai/ --acl public-read
