#!/bin/bash

cd $1

OUT=.htaccess
echo "ErrorDocument 404 https://zh.gluon.ai/404.html" >$OUT
echo "RewriteEngine On" >>$OUT
echo "RewriteCond %{SERVER_PORT} 80" >>$OUT
echo "RewriteRule ^(.*)$ https://zh.gluon.ai/\$1 [R,L]" >>$OUT
for f in chapter*/*; do
    if [[ $f == *"index"* ]]; then
        continue
    fi
    echo "Redirect /$(basename $f) /$f" >>$OUT
done

echo "Redirect /chapter_convolutional-neural-networks/kaggle-gluon-cifar10.html /chapter_computer-vision/kaggle-gluon-cifar10.html">>$OUT
