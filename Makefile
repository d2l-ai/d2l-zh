all: html

build/%.ipynb: %.md build/build.yml utils.py
	@mkdir -p $(@D)
	cd $(@D); python ../md2ipynb.py ../../$< ../../$@

build/%.md: %.md
	@mkdir -p $(@D)
	@cp $< $@

MARKDOWN = $(wildcard */index.md) chapter_crashcourse/introduction.md chapter_appendix/aws.md
NOTEBOOK = $(filter-out $(MARKDOWN), $(wildcard chapter*/*.md))

OBJ = $(patsubst %.md, build/%.md, $(MARKDOWN)) \
	$(patsubst %.md, build/%.ipynb, $(NOTEBOOK))

ORIGN_DEPS = $(wildcard img/* data/*) environment.yml utils.py README.md
DEPS = $(patsubst %, build/%, $(ORIGN_DEPS))

PKG = build/_build/html/gluon_tutorials_zh.tar.gz build/_build/html/gluon_tutorials_zh.zip

pkg: $(PKG)

build/_build/html/gluon_tutorials_zh.zip: $(OBJ) $(DEPS)
	cd build; zip -r $(patsubst build/%, %, $@ $(DEPS)) chapter*

build/_build/html/gluon_tutorials_zh.tar.gz: $(OBJ) $(DEPS)
	cd build; tar -zcvf $(patsubst build/%, %, $@ $(DEPS)) chapter*

build/%: %
	@mkdir -p $(@D)
	@cp -r $< $@

html: $(DEPS) $(OBJ)
	make -C build html
	bash build/htaccess.sh build/_build/html/
	cp tencent1668843323268181422.txt build/_build/html/tencent1668843323268181422.txt

TEX=build/_build/latex/gluon_tutorials_zh.tex

SVG=$(wildcard img/*.svg)
GIF=$(wildcard img/*.gif)

build/_build/latex/%.pdf: img/%.svg
	@mkdir -p $(@D)
	rsvg-convert -f pdf -o $@ $<

build/_build/latex/%_00.pdf: img/%_00.pdf
	@mkdir -p $(@D)
	cp $< $@

PDFIMG = $(patsubst img/%.svg, build/_build/latex/%.pdf, $(SVG)) \
	$(patsubst img/%.gif, build/_build/latex/%_00.pdf, $(GIF))

pdf: $(DEPS) $(OBJ) $(PDFIMG)
	@echo $(PDFIMG)
	make -C build latex
	sed -i s/\.svg/\.pdf/ $(TEX)
	sed -i s/\}\.gif/\_00\}.pdf/ $(TEX)
	sed -i s/{tocdepth}{0}/{tocdepth}{1}/ $(TEX)
	sed -i s/{\\\\releasename}{发布}/{\\\\releasename}{}/ $(TEX)
	sed -i s/{OriginalVerbatim}\\\[commandchars=\\\\\\\\\\\\{\\\\}\\\]/{OriginalVerbatim}\\\[commandchars=\\\\\\\\\\\\{\\\\},formatcom=\\\\footnotesize\\\]/ $(TEX)
	sed -i s/\\\\usepackage{geometry}/\\\\usepackage[paperwidth=187mm,paperheight=235mm,left=20mm,right=20mm,top=20mm,bottom=15mm,includefoot]{geometry}/ $(TEX)
	cd build/_build/latex && \
	buf_size=10000000 xelatex gluon_tutorials_zh.tex && \
	buf_size=10000000 xelatex gluon_tutorials_zh.tex

clean:
	rm -rf build/chapter* $(DEPS) $(PKG)
