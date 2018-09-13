#!/bin/bash

pre() {
	echo "Pre-processing markdown files in source lauguage.";
	for f in chapter*/*.md; do
		echo $f
		sed -i s/\.python\ \.input/\.python-\.input/g $f
		sed -i s/\.input\ \ n=/\.input-n=/g $f
		sed -i s/\</%%%less-than%%%/g $f
		sed -i s/\&/%%%ampersand%%%/g $f
	done
}

extract() {
	echo "Convert markdown files into xliff (in source language) and skeleton files.";
	BSL="bookSrcLang"
	[ -e $BSL ] && rm -rf $BSL
	mkdir -p $BSL
	for f in chapter*/*.md; do
		echo $f
		xlf="${f%%.*}.xlf"
		sklmd="${f%%.*}.skl.md"
		./md2xliff/bin/extract $f $xlf $sklmd 'zh-CN' 'en-US'
		# Generate bookSrcLang that contains only xlf files.
		dir=$(dirname "$f")
		mkdir -p $BSL/$dir
		base=$(basename $f)
		xlf_base="${base%%.*}.xlf"
		cp $xlf $BSL/$dir/$xlf_base
	done
}

reconstruct() {
	echo "Convert xliff (in target language) and skeleton files into markdown files.";
	BTL="bookTgtLang"
	for f in chapter*/*.xlf; do
		echo $f
		# Load xlf files from translated dir.
		cp $BTL/$f $f
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
		sed -i s/%%%ampersand%%%/\\\&/g $f
	done
}

"$@"
