tgt=build/2020CS10869_aniruddha_deb

submission: Q
	cp tex/writeup.pdf $(tgt)
	(cd build && zip -r 2020CS10869_aniruddha_deb.zip 2020CS10869_aniruddha_deb)

dir:
	mkdir -p build
	mkdir -p $(tgt)

Q: dir
	cp -r Q* $(tgt)
	find $(tgt)/Q* -type f -not -name "*.py" -delete
	rm -rf $(tgt)/Q*/.ipynb*

clean:
	rm -rf build/*
