#!/bin/bash                                                                                                   

# Prerequisite: pip install nblint

OUT=outlint

[ -e $OUT ] && rm $OUT

for f in build/chapter*/*.ipynb; do
	echo '===' $f
	echo '===' $f >> $OUT
	nblint --linter pyflakes $f >> $OUT 
	nblint $f >> $OUT
done

# E302 expected 2 blank lines, found 1
# E305 expected 2 blank lines after class or function definition, found 1
# E402 module level import not at top of file
# E703 statement ends with a semicolon
# E741 ambiguous variable name
IGNORE=( 'E302' 
		 'E305'
		 'E402' 
		 'E703'
		 'E741' )

for ign in "${IGNORE[@]}"; do
	sed -i /$ign/d $OUT
done
