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

# avoid to put the downloaed data into the notebook package
mv build/data build/data-bak
make pkg
# backup build/data to avoid download the dataset each time and put the
rm -rf build/data
mv build/data-bak build/data


make pdf
cp build/_build/latex/gluon_tutorials_zh.pdf build/_build/html/

aws s3 sync --delete build/_build/html/ s3://zh.gluon.ai/ --acl public-read
