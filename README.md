# MSc Thesis – <working title>


## Requirements

| Platform | TeX distribution | Extras          |
|----------|------------------|-----------------|
| Windows  | TeX Live 2024+   | `latexmk`, `biber` |
| WSL      | `texlive-full`   | MS Core Fonts (`ttf-mscorefonts-installer`) |
| macOS    | MacTeX 2024      | — |


## Rough install guide
You will be prompted to accept some terms and conditions, press Tab to select *OK* then enter to accept.
```bash
sudo apt update && sudo apt install texlive-full latexmk biber ttf-mscorefonts-installer
fc-cache -fv    # refresh fontconfig so XeLaTeX can see Georgia/Arial
```

Then, use the _LaTeX Workshop_ extension to compile. 