MD="booksrc" 

[ -e $MD ] && rm -rf $MD
mkdir $MD

cp index.md $MD/
cp -R img $MD/
for d in chapter*; do
	cp -R $d $MD/
done

for f in $MD/chapter*/*ipynb; do
	base=$(basename $f)
	jupyter nbconvert --to markdown $f --output "${base%%.*}.md" 
	rm $f
done

tar -zcvf $MD
