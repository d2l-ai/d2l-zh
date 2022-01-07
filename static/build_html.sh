#!/bin/bash

set -e

rm -rf _build/rst _build/html
d2lbook build rst --tab all
cp static/frontpage/frontpage.html _build/rst_all/
d2lbook build html --tab all
cp -r static/frontpage/_images/* _build/html/_images/

for fn in `find _build/html/_images/ -iname '*.svg' `; do
    if [[ $fn == *'qr_'* ]] ; then # || [[ $fn == *'output_'* ]]
        continue
    fi
    # rsvg-convert installed on ubuntu changes unit from px to pt, so evening no
    # change of the size makes the svg larger...
    rsvg-convert -z 1 -f svg -o tmp.svg $fn
    mv tmp.svg $fn
done

# Add SageMaker Studio Lab buttons
for f in _build/html/chapter*/*.html; do
    sed -i s/Open\ the\ notebook\ in\ Colab\<\\\/div\>\<\\\/div\>\<\\\/div\>\<\\\/h1\>/Open\ the\ notebook\ in\ Colab\<\\\/div\>\<\\\/div\>\<\\\/div\>\<a\ href=\"https:\\\/\\\/studiolab.sagemaker.aws\\\/import\\\/github\\\/d2l-ai\\\/d2l-pytorch-sagemaker-studio-lab\\\/blob\\\/main\\\/GettingStarted-D2L.ipynb\"\ onclick=\"captureOutboundLink\\\(\'https\:\\\/\\\/studiolab.sagemaker.aws\\\/import\\\/github\\\/d2l-ai\\\/d2l-pytorch-sagemaker-studio-lab\\\/blob\\\/main\\\/GettingStarted-D2L.ipynb\'\\\)\;\ return\ false\;\"\>\ \<button\ style=\"float\:right\",\ id=\"SageMaker\_Studio\_Lab\"\ class=\"mdl-button\ mdl-js-button\ mdl-button--primary\ mdl-js-ripple-effect\"\>\ \<i\ class=\"\ fas\ fa-external-link-alt\"\>\<\\\/i\>\ SageMaker\ Studio\ Lab\ \<\\\/button\>\<\\\/a\>\<div\ class=\"mdl-tooltip\"\ data-mdl-for=\"SageMaker\_Studio\_Lab\"\>\ Open\ the\ notebook\ in\ SageMaker\ Studio\ Lab\<\\\/div\>\<\\\/h1\>/g $f
done
