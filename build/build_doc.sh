#!/bin/bash
#
# Build and publish all docs into Pulish all notebooks to mxnet.
set -x
set -e

# prepare the env
conda env update -f build/build.yml
source activate gluon_zh_docs

pip list

make html

rm -rf build/data
make pkg

make pdf
cp build/_build/latex/gluon_tutorials_zh.pdf build/_build/html/

aws s3 sync --delete build/_build/html/ s3://zh.gluon.ai/ --acl public-read
