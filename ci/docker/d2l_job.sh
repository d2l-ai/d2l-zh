#!/bin/bash

date
echo "Args: $@"
env
echo "jobId: $AWS_BATCH_JOB_ID"
echo "jobQueue: $AWS_BATCH_JQ_NAME"
echo "computeEnvironment: $AWS_BATCH_CE_NAME"

SOURCE_REF=$1
WORK_DIR=$2
COMMAND=$3
############### NOT USED ATM ##################
SAVED_OUTPUT=$4
SAVE_PATH=$5
###############################################
REMOTE=$6
SAFE_TO_USE_SCRIPT=$7
ORIGINAL_REPO=${8:-'d2l-zh'}
# TODO @anirudhdagar: hardcode ORIGINAL_ORG
# Avoid ability to change org by restricting
# job definition arguments defined in d2l-infra
# This is only changed for testing purposes
ORIGINAL_ORG=${9:-'d2l-ai'}


# Copy the workflow from master branch
git clone https://github.com/"$ORIGINAL_ORG"/"$ORIGINAL_REPO".git

WORKFLOW_SCRIPTS="$ORIGINAL_REPO"/.github/workflow_scripts
if [ -d "$WORKFLOW_SCRIPTS" ]; then
    cp -R "$ORIGINAL_REPO"/.github/workflow_scripts .
fi

cd "$ORIGINAL_REPO"

if [ ! -z $REMOTE ]; then
    git remote set-url origin $REMOTE
fi

git fetch origin $SOURCE_REF:working
git checkout working

# Reset modification times for all notebooks using git-timesync
# We use this to make sure d2lbook build eval caching is valid
# even after cloning the repo for each run
# Modification times for original repo files are corrected and are now
# good for comparing with modification times of build files coming
# from the S3 bucket
git timesync *.md **/*.md

# If not safe to use script, we overwrite with the script from master branch
TRUE=true
if [[ ${SAFE_TO_USE_SCRIPT,,} != ${TRUE,,} ]]; then
    if [ -d ../workflow_scripts ]; then
        rm -rf .github/workflow_scripts
        mv ../workflow_scripts .github/
    else
        echo Not safe to use user provided script, and could not find script from master branches
        exit 1
    fi
fi

cd $WORK_DIR
/bin/bash -o pipefail -c "eval $COMMAND"
COMMAND_EXIT_CODE=$?

exit $COMMAND_EXIT_CODE
