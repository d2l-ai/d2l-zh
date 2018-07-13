#!/bin/bash

pre() {
	echo "Pre-processing markdown files in source lauguage.";
	for f in chapter*/*.md; do
		echo $f
		sed -i s/\.python\ \.input/\.python-\.input/g $f
		sed -i s/\.input\ \ n=/\.input-n=/g $f
		sed -i s/\</%%%less-than%%%/g $f
	done
}

extract() {
	echo "Convert markdown files into xliff (in source language) and skeleton files.";
	for f in chapter*/*.md; do
		echo $f
		./md2xliff/bin/extract $f
	done
}

reconstruct() {
	echo "Convert xliff (in target language) and skeleton files into markdown files.";
	for f in chapter*/*.xlf; do
		echo $f
		md="${f%%.*}.md"
		sklmd="${f%%.*}.skl.md"
		./md2xliff/bin/xliff-reconstruct $f $sklmd $md
	#rm $f
	#rm $sklmd
	done
}

post() {
	echo "Post-processing markdown files in target language.";
	for f in chapter*/*.md; do
		echo $f
		sed -i s/\.python-\.input/\.python\ \.input/g $f
		sed -i s/\.input-n=/\.input\ \ n=/g $f
		sed -i s/%%%less-than%%%/\</g $f
	done
}

"$@"
