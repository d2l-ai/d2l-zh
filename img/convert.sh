ls *.pdf | while read f; do pdf2svg $f ${f%.pdf}.svg; done
