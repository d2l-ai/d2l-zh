#!/bin/bash
set -e

conda env update -f build/env.yml
source activate d2l-zh-build

pip list

rm -rf build/_build/

make html EVAL=0
