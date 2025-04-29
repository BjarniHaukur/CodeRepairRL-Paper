# MSc Thesis – <working title>



## Requirements

| Platform | TeX distribution | Extras          |
|----------|------------------|-----------------|
| Windows  | TeX Live 2024+   | `latexmk`, `biber` |
| WSL      | `texlive-full`   | MS Core Fonts (`ttf-mscorefonts-installer`) |
| macOS    | MacTeX 2024      | — |

 _LaTeX Workshop_ is recommended.

## Rough install guide
```bash
sudo apt update && sudo apt install texlive-full latexmk biber ttf-mscorefonts-installer
fc-cache -fv    # refresh fontconfig so XeLaTeX can see Georgia/Arial
```

## Quick start

```bash
make pdf        # one-shot build → build/main.pdf
make watch      # continuous build; recompiles on save
make clean      # remove intermediates, keep build/main.pdf
make distclean  # remove build/ and the PDF
```

## Declutter
This latex setup creates a ton of intermediary / temp files, I recommend putting this in your user / workspace settings.json
```json
{
    "files.exclude": {
        "**/*.aux": true,
        "*main.bbl": true,
        "*main.bcf": true,
        "*main.blg": true,
        "*main.fdb_latexmk": true,
        "*main.fls": true,
        "*main.log": true,
        "*main.out": true,
        "*main.run.xml": true,
        "*main.synctex.gz": true,
        "*main.toc": true,
        "*main.xdv": true,
    }
}
```