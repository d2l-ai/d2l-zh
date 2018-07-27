#!/bin/bash                                                                                                   

# Prerequisite: pip install nblint
for f in build/chapter*/*.ipynb; do
	echo '===' $f
	nblint --linter pyflakes $f 
	nblint $f 
done
