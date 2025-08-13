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


## macOS setup

### 1) Install a TeX distribution

- Recommended (full, easiest):
  ```bash
  brew install --cask mactex
  ```
  After install, ensure TeX binaries are on PATH (usually automatic). If needed, add to `~/.zshrc`:
  ```bash
  echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc && source ~/.zshrc
  ```

- Alternative (minimal):
  ```bash
  brew install --cask basictex
  sudo tlmgr update --self
  # You will likely need many packages; at minimum:
  sudo tlmgr install latexmk biber biblatex biblatex-ieee fontspec csquotes tcolorbox tocloft sectsty svg yhmath siunitx booktabs multirow tikzlings
  ```
  Note: BasicTeX may require additional packages beyond the above; using full MacTeX avoids chasing dependencies.

### 2) Compile with XeLaTeX (required for system fonts)

From the repository root:
```bash
latexmk -xelatex -usebiber -synctex=1 -interaction=nonstopmode -file-line-error main.tex
```
If you prefer VS Code, select the XeLaTeX recipe in the LaTeX Workshop extension.

Georgia and Arial are bundled with macOS, so no extra font install is needed.

#### LaTeX Workshop (VS Code)

- Works out-of-the-box with MacTeX if `/Library/TeX/texbin` is on PATH. If VS Code doesn't see the binaries, add that path to your shell and relaunch VS Code.
- Ensure the recipe uses XeLaTeX + biber. You can add this to your VS Code settings (User or Workspace `settings.json`):

```json
{
  "latex-workshop.latex.tools": [
    {
      "name": "latexmk",
      "command": "latexmk",
      "args": [
        "-xelatex",
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-usebiber",
        "%DOC%"
      ]
    }
  ],
  "latex-workshop.latex.recipes": [
    { "name": "latexmk (xelatex + biber)", "tools": ["latexmk"] }
  ]
}
```

##### Fix LaTeX formatting (latexindent) on macOS

If formatting fails with errors like missing `YAML::Tiny` or `File::HomeDir`, install the required Perl modules and ensure VS Code uses them:

1) Install Perl and cpanminus (Apple Silicon shown; for Intel, replace `/opt/homebrew` with `/usr/local`):
```bash
brew install perl cpanminus
echo 'export PATH="/opt/homebrew/opt/perl/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

2) Install the modules latexindent needs:
```bash
cpanm YAML::Tiny File::HomeDir Log::Log4perl Log::Dispatch Unicode::LineBreak Unicode::GCString
```

3) Make LaTeX Workshop see Homebrew Perl and TeX in VS Code `settings.json`:
```json
{
  "latex-workshop.environment": {
    "PATH": "/opt/homebrew/opt/perl/bin:/Library/TeX/texbin:${env:PATH}"
  }
}
```

4) Restart VS Code. If needed, temporarily disable format on save:
```json
{
  "[latex]": { "editor.formatOnSave": false }
}
```