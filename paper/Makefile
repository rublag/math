.PHONY: all paper

all: paper

paper:
	mkdir -p ../build/paper
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/paper paper.tex
	cp ../build/paper/paper.pdf ../build/paper.pdf
