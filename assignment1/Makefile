tgt=build/2020CS10869_aniruddha_deb

submission: Q
	cp tex/writeup.pdf $(tgt)
	(cd build && zip -r 2020CS10869_aniruddha_deb.zip 2020CS10869_aniruddha_deb)

dir:
	mkdir -p build
	mkdir -p $(tgt)

Q: dir
	mkdir -p $(tgt)/Q1
	cp Q1/*.py $(tgt)/Q1
	cp -r Q1/plots $(tgt)/Q1
	mkdir -p $(tgt)/Q2
	cp Q2/*.py $(tgt)/Q2
	cp -r Q2/plots $(tgt)/Q2
	mkdir -p $(tgt)/Q3
	cp Q3/*.py $(tgt)/Q3
	cp -r Q3/plots $(tgt)/Q3
	mkdir -p $(tgt)/Q4
	cp Q4/*.py $(tgt)/Q4
	cp -r Q4/plots $(tgt)/Q4

clean:
	rm -rf build/*
