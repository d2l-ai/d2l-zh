MD="mdd" 

[ -e $MD ] && rm -rf $MD
mkdir $MD

cp index.md $MD/
cp -R img $MD/
for f in chapter*/*; do
	dir=$(dirname "$f")
	if [ "${f##*.}" = "md" ] || [ "${f##*.}" = "ipynb" ]; then
		mkdir -p $MD/$dir
		cp $f $MD/$f
	fi
done

for f in $MD/chapter*/*ipynb; do
    base=$(basename $f)
    jupyter nbconvert --to markdown $f --output "${base%%.*}.md" 
    rm $f
done

zip -r "$MD.zip" $MD 
[ -e $MD ] && rm -rf $MD 
