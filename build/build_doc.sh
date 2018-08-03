#!/bin/bash
# Build and publish all docs into Pulish all notebooks to mxnet.
set -x
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
cd ..

# prepare the env
conda env update -f build/build.yml
conda activate gluon_zh_docs

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

[ -e build/_build/latex/gluon_tutorials_zh.aux ] && rm build/_build/latex/gluon_tutorials_zh.aux
[ -e build/_build/latex/gluon_tutorials_zh.idx ] && rm build/_build/latex/gluon_tutorials_zh.idx

# FOR 1.0
cp build/_build/html/gluon_tutorials_zh.tar.gz build/_build/html/gluon_tutorials_zh-1.0.tar.gz
cp build/_build/html/gluon_tutorials_zh.zip build/_build/html/gluon_tutorials_zh-1.0.zip


cd build

bash ipynb2mdd.sh
cp mdd.zip _build/html/mdd.zip

cd ..

aws s3 sync --delete build/_build/html/ s3://zh.gluon.ai/ --acl public-read
