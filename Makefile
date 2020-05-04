.PHONY: all paper clean

all: paper

clean:
	rm -rf build

paper:
	mkdir -p build
	$(MAKE) -C paper
	cp build/paper.pdf ./
