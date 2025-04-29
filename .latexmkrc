# .latexmkrc  –– run XeLaTeX -> Biber -> XeLaTeX ×2
$pdflatex   = 'xelatex -synctex=1 -interaction=nonstopmode %O %S';
$biber      = 'biber %O %S';
$bibtex_use = 2;          # tell latexmk we’re using biber
$pdf_previewer = 'start'; # opens PDF in default viewer on Windows
$clean_ext .= " acn acr alg glg glo ist loa loc lol lot out snm synctex* toc xnav";