#!/bin/bash
set -e

# Clean build/chapter*/*ipynb and build/chapter*/*md that are no longer needed.
cd build
for ch in chapter*; do
    if ! [ -e "../$ch" ]; then
        rm -rf $ch
    else
        shopt -s nullglob
        for f in $ch/*.md $ch/*.ipynb; do
            echo $f
            base=$(basename $f)
            md=${base%%.*}.md
            if ! [ -e "../$ch/$md" ]; then
                rm $f
            fi
        done
    fi
done
# Clean images that are no longer needed.
shopt -s nullglob
for f in img/*.svg img/*.jpg img/*.png; do
    echo $f
    if ! [ -e "../$f" ]; then
        rm $f
    fi
done
cd ..


git submodule update --init
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

conda env update -f build/env.yml
conda activate d2l-zh-build

pip list

rm -rf build/_build/

make html
