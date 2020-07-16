#!/bin/bash

if [ $# -ne 2 ]; then
   echo "usage:"
   echo "  $0 store dir"
   echo "  $0 restore dir"
   exit -1
fi

cmd=$1
dir=$2
saved_dir="${dir///data/_data}"

if [ $cmd == "store" ]; then
    if [ -e $dir ]; then
       rm -rf $saved_dir
       mv $dir $saved_dir
    fi
    echo "Saved $dir to $saved_dir"
elif [ $cmd == "restore" ]; then
    if [ -e $saved_dir ]; then
       rm -rf $dir
       mv $saved_dir $dir
    fi
    echo "Restored $dir from $saved_dir"
else
    echo "unknown command $1, should be either store or restore"
    exit -1
fi
