all: html

build/%.ipynb: %.md build/env.yml $(wildcard d2lzh/*)
	@mkdir -p $(@D)
	cd $(@D); python ../utils/md2ipynb.py ../../$< ../../$@

build/%.md: %.md
	@mkdir -p $(@D)
	@cp $< $@

MARKDOWN = $(wildcard */index.md)
NOTEBOOK = $(filter-out $(MARKDOWN), $(wildcard chapter*/*.md))

OBJ = $(patsubst %.md, build/%.md, $(MARKDOWN)) \
	$(patsubst %.md, build/%.ipynb, $(NOTEBOOK))

FRONTPAGE_DIR = img/frontpage
FRONTPAGE = $(wildcard $(FRONTPAGE_DIR)/*)
FRONTPAGE_DEP = $(patsubst %, build/%, $(FRONTPAGE))

IMG_NOTEBOOK = $(filter-out $(FRONTPAGE_DIR), $(wildcard img/*))
ORIGIN_DEPS = $(IMG_NOTEBOOK) $(wildcard data/* d2lzh/*) environment.yml README.md
DEPS = $(patsubst %, build/%, $(ORIGIN_DEPS))

PKG = build/_build/html/d2l-zh.zip

pkg: $(PKG)

build/_build/html/d2l-zh.zip: $(OBJ) $(DEPS)
	cd build; zip -r $(patsubst build/%, %, $@ $(DEPS)) chapter*/*md chapter*/*ipynb

# Copy XX to build/XX if build/XX is depended (e.g., $(DEPS))
build/%: %
	@mkdir -p $(@D)
	@cp -r $< $@

html: $(DEPS) $(FRONTPAGE_DEP) $(OBJ)
	make -C build html
	python build/utils/post_html.py
	cp -r img/frontpage/ build/_build/html/_images/
	# Enable horitontal scrollbar for wide code blocks
	sed -i s/white-space\:pre-wrap\;//g build/_build/html/_static/sphinx_materialdesign_theme.css

TEX=build/_build/latex/d2l-zh.tex

build/_build/latex/%.pdf: img/%.svg
	@mkdir -p $(@D)
	rsvg-convert -f pdf -z 0.80 -o $@ $<

SVG=$(wildcard img/*.svg)

PDFIMG = $(patsubst img/%.svg, build/_build/latex/%.pdf, $(SVG))

pdf: $(DEPS) $(OBJ) $(PDFIMG)
	@echo $(PDFIMG)
	make -C build latex
	sed -i s/\\.svg/.pdf/g ${TEX}
	sed -i s/\}\\.gif/\_00\}.pdf/g $(TEX)
	sed -i s/{tocdepth}{0}/{tocdepth}{1}/g $(TEX)
	sed -i s/{\\\\releasename}{发布}/{\\\\releasename}{}/g $(TEX)
	sed -i s/{OriginalVerbatim}\\\[commandchars=\\\\\\\\\\\\{\\\\}\\\]/{OriginalVerbatim}\\\[commandchars=\\\\\\\\\\\\{\\\\},formatcom=\\\\footnotesize\\\]/g $(TEX)
	sed -i s/\\\\usepackage{geometry}/\\\\usepackage[paperwidth=187mm,paperheight=235mm,left=20mm,right=20mm,top=20mm,bottom=15mm,includefoot]{geometry}/g $(TEX)
	# Allow figure captions to include space and autowrap
	sed -i s/Ⓐ/\ /g ${TEX}
	# Remove un-translated long table descriptions
	sed -i /\\\\multicolumn{2}{c}\%/d $(TEX)
	sed -i /\\\\sphinxtablecontinued{Continued\ on\ next\ page}/d $(TEX)
	sed -i /{\\\\tablename\\\\\ \\\\thetable{}\ --\ continued\ from\ previous\ page}/d $(TEX)
	sed -i s/\\\\maketitle/\\\\maketitle\ \\\\pagebreak\\\\hspace{0pt}\\\\vfill\\\\begin{center}本书稿为测试版本（\ 生成日期：\\\\zhtoday\ ）。\\\\\\\\\ 访问\\\\url{https:\\/\\/zh.d2l.ai}，获取本书的最新版本或正式版本。\\\\end{center}\\\\vfill\\\\hspace{0pt}\\\\pagebreak/g $(TEX)

	python build/utils/post_latex.py zh

	cd build/_build/latex && \
	bash ../../utils/convert_output_svg.sh && \
	buf_size=10000000 xelatex d2l-zh.tex && \
	buf_size=10000000 xelatex d2l-zh.tex

clean:
	rm -rf build/chapter* build/_build build/img build/data build/environment.yml build/README.md $(PKG)
