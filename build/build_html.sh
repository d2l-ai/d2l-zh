#!/bin/bash
set -e
git submodule update --init
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

conda env update -f build/env.yml
conda activate d2l-zh-build

pip list

make clean

rm -rf build/_build/

make html
