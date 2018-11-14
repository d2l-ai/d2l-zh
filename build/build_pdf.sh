#!/bin/bash
set -e

conda activate d2l-zh-build

make pdf
cp build/_build/latex/d2l-zh.pdf build/_build/html/
