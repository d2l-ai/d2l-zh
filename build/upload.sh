#!/bin/bash
set -e
set -x

conda activate d2l-zh-build

BUCKET=s3://zh.diveintodeeplearning.org
# BUCKET=s3://diveintodeeplearning-staging

DIR=build/_build/html/

#find $DIR \( -iname '*.css' -o -iname '*.js' \) -exec gzip -9 -n {} \; -exec mv {}.gz {} \;

#aws s3 sync --exclude '*.*' --include '*.css' \
#      --content-type 'text/css' \
#      --content-encoding 'gzip' \
#      --acl 'public-read' \
#      $DIR $BUCKET

aws s3 sync --exclude '*.*' --include '*.woff' --include '*.woff2' \
      --expires "$(date -d '+24 months' --utc +'%Y-%m-%dT%H:%M:%SZ')" \
      --acl 'public-read' \
      $DIR $BUCKET

#aws s3 sync --exclude '*.*' --include '*.js' \
#      --content-type 'application/javascript' \
#      --content-encoding 'gzip' \
#      --acl 'public-read' \
#      $DIR $BUCKET

aws s3 sync --delete $DIR $BUCKET --acl 'public-read'
