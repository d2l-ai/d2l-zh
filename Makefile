all: html

build/%.ipynb: %.md environment.yml utils.py
	@mkdir -p $(@D)
	python build/md2ipynb.py $< $@

build/%.md: %.md
	@mkdir -p $(@D)
	@cp $< $@

MARKDOWN = $(wildcard chapter_preface/*.md */index.md)
NOTEBOOK = $(filter-out $(MARKDOWN), $(wildcard chapter*/*.md))

OBJ = $(patsubst %.md, build/%.md, $(MARKDOWN)) \
	$(patsubst %.md, build/%.ipynb, $(NOTEBOOK))

build/img:
	rsync -rupE img build/

build/data:
	rsync -rupE data build/

.PHONY: build/img build/data

html: $(OBJ) build/img build/data
	make -C build html

clean:
	rm -rf build/chapter* build/img build/data
