all: html

build/%.ipynb: %.md environment.yml utils.py
	@mkdir -p $(@D)
	cd build; python md2ipynb.py ../$< ../$@

build/%.md: %.md
	@mkdir -p $(@D)
	@cp $< $@

MARKDOWN = $(wildcard chapter_preface/*.md */index.md)
NOTEBOOK = $(filter-out $(MARKDOWN), $(wildcard chapter*/*.md))

OBJ = $(patsubst %.md, build/%.md, $(MARKDOWN)) \
	$(patsubst %.md, build/%.ipynb, $(NOTEBOOK))

DEPS = build/img build/data build/environment.yml build/utils.py build/LICENSE build/README.md

build/%: %
	@cp $< $@

build/img:
	rsync -rupE img build/

build/data:
	rsync -rupE data build/


html: $(OBJ) $(DEPS)
	make -C build html

clean:
	rm -rf build/chapter* $(DEPS)
