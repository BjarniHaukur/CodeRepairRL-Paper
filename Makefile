# ---------- configurable bits ----------
DOC=main
LATEXMKOPTS = -xelatex -synctex=1 -interaction=nonstopmode
# ---------------------------------------

.PHONY: all pdf biber watch clean distclean

all: pdf

pdf: $(DOC).pdf

$(DOC).pdf: $(DOC).tex
	latexmk $(LATEXMKOPTS) $(DOC).tex

biber:
	biber $(DOC)

watch:
	latexmk -pvc -xelatex -synctex=1 -interaction=nonstopmode $(DOC).tex

clean:
	latexmk -C
	@rm -f *.aux *.log *.bbl *.bcf *.blg *.run.xml *.toc *.out *.fls *.fdb_latexmk *.lof *.lot *.dvi *.idx *.ilg *.ind *.synctex.gz

distclean: clean
	@rm -f $(DOC).pdf
