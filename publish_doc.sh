#!/bin/bash
#
# Build and publish all docs into Pulish all notebooks to mxnet.
set -x
set -e

# prepare the env
conda env update -f environment.yml
source activate gluon_zh_docs

make html

rm -rf ~/zh/latest
mv _build/html ~/zh/latest
