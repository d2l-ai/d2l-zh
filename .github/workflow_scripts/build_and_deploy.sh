#!/bin/bash

# Move all scripts related to html here!

set -ex

REPO_NAME="$1"  # Eg. 'd2l-zh'
TARGET_BRANCH="$2" # Eg. 'master' ; if PR raised to master
JOB_NAME="$3" # Eg. 'd2l-zh/master' or 'd2l-zh/PR-2453/21be1a4'
LANG="$4" # Eg. 'en','zh' etc.
CACHE_DIR="$5"  # Eg. 'ci_cache_pr' or 'ci_cache_push'

pip3 install d2l==0.17.6
mkdir _build

source $(dirname "$0")/utils.sh

# Move aws copy commands for cache restore outside
measure_command_time "aws s3 sync s3://preview.d2l.ai/"$CACHE_DIR"/"$REPO_NAME"-"$TARGET_BRANCH"/_build _build --delete --quiet --exclude 'eval*/data/*'"

# Build D2L Website
./.github/workflow_scripts/build_html.sh

# Build PDFs
d2lbook build pdf
d2lbook build pdf --tab pytorch


# Check if the JOB_NAME is "$REPO_NAME/release"
if [[ "$JOB_NAME" == "$REPO_NAME/release" ]]; then

  # Setup D2L Bot
  source $(dirname "$0")/setup_git.sh
  setup_git

  d2lbook build pkg
  d2lbook deploy html pdf pkg colab sagemaker slides --s3 "s3://${LANG}-v2.d2l.ai/"

else
  # Run d2lbook preview deployment
  d2lbook deploy html pdf --s3 "s3://preview.d2l.ai/${JOB_NAME}/"
fi

# Move aws copy commands for cache store outside
measure_command_time "aws s3 sync _build s3://preview.d2l.ai/"$CACHE_DIR"/"$REPO_NAME"-"$TARGET_BRANCH"/_build --acl public-read --quiet --exclude 'eval*/data/*'"
