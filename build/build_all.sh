#!/bin/bash
set -ex

tik=$(date +%s)

[ -e build/data-bak ] && rm -rf build/data-bak

# Clean build/chapter*/*ipynb and build/chapter*/*md that are no longer needed.
cd build
for ch in chapter*; do
    if ! [ -e "../$ch" ]; then
        rm -rf $ch
    else
        shopt -s nullglob
        for f in $ch/*.md $ch/*.ipynb; do
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

make pdf
cp build/_build/latex/d2l-zh.pdf build/_build/html/

[ -e build/_build/latex/d2l-zh.aux ] && rm build/_build/latex/d2l-zh.aux
[ -e build/_build/latex/d2l-zh.idx ] && rm build/_build/latex/d2l-zh.idx

# avoid putting data downloaded by scripts into the notebook package
mv build/data build/data-bak
make pkg
# backup build/data to avoid download the dataset each time and put the
rm -rf build/data
mv build/data-bak build/data

# For 1.0
cp build/_build/html/d2l-zh.zip build/_build/html/d2l-zh-1.0.zip

# Time it
tok=$(date +%s)
runtime=$((tok-tik))
convertsecs() {
	((h=${1}/3600))
	((m=(${1}%3600)/60))
	((s=${1}%60))
	printf "%02d:%02d:%02d\n" $h $m $s
}
echo $(convertsecs $runtime)
