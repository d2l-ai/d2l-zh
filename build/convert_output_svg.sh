set -x
set -e
for f in *.svg; do
    rsvg-convert -f pdf -z 0.80 -o ${f%.svg}.pdf $f
done
