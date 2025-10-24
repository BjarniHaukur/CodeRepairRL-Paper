# KTH Thesis Presentation Slides

Professional Marp-based presentation slides for the thesis defense, covering chapters 0-4 with full KTH Royal Institute of Technology branding.

## Quick Start

### Prerequisites

Install Marp CLI globally:

```bash
npm install -g @marp-team/marp-cli
```

**Or** use the VS Code extension (recommended for development):
- Install "Marp for VS Code" from the marketplace
- Open `presentation.md`
- Click the preview icon or press `Ctrl+K V`

### Building Slides

#### HTML (recommended for presentation)

```bash
cd slides
marp presentation.md --html --theme kth-theme.css -o presentation.html
```

The `--html` flag is **required** to enable iframe embedding for Sankey diagrams.

#### PDF Export

```bash
marp presentation.md --html --theme kth-theme.css --allow-local-files -o presentation.pdf
```

**Note**: PDF export may not render iframes correctly. Use HTML for the actual presentation.

#### PowerPoint

```bash
marp presentation.md --theme kth-theme.css -o presentation.pptx
```

### Live Development

Watch mode with auto-rebuild:

```bash
marp -w --html --theme kth-theme.css presentation.md
```

Serve with live preview:

```bash
marp -s -w --html --theme kth-theme.css presentation.md
```

Then open http://localhost:8080 in your browser.

## Presentation Structure

### Duration & Pacing
- **Target**: 20-minute thesis defense
- **Slides**: ~40 slides (30-35 seconds per slide average)
- **Flow**: Frontmatter (2min) → Background (5min) → Method (5min) → Implementation (5min) → Buffer (3min)

### Content Coverage

#### Part 1: Front Matter & Introduction (Slides 1-10)
- KTH-branded title slide with thesis information
- Research scope, motivation, and problem statement
- Three research questions (harness, performance, multilingual)
- Key contributions and delimitations

#### Part 2: Background & Related Work (Slides 11-20)
- Evolution of automated program repair (pre-LLM → LLM-based)
- Scaffold taxonomy: scaffold-free, agentless, agentic
- RL algorithm progression: PPO → GRPO → GSPO
- Comparison table and related work positioning

#### Part 3: Methodology (Slides 21-28)
- Nano agent architecture and philosophy
- Observation and action spaces with examples
- Training data curriculum (1,000 tasks)
- Reward design and GSPO training objective

#### Part 4: Implementation (Slides 29-38)
- Online RL infrastructure with live weight sync
- Async episode collection and NCCL communication
- SLURM orchestration on academic clusters
- Compute optimizations (LoRA, ZeRO-2, gradient checkpointing)
- **Interactive Sankey diagrams** (early and late training)

## Custom KTH Theme

The `kth-theme.css` file implements KTH Royal Institute of Technology branding:

### Colors
- **Primary Blue**: `#1954a6` (headers, accents)
- **Light Blue**: `#3a7bc8` (gradients, secondary)
- **Dark Blue**: `#0e3b6f` (emphasis)
- **Gray**: `#65656c` (body text, footers)

### Typography
- **Headers**: Arial (sans-serif) - matches KTH official style
- **Body**: Georgia (serif) - matches thesis body text
- **Code**: Courier New (monospace)

### Slide Types

**Lead slides** (`<!-- _class: lead -->`):
- Full-screen gradient background (blue)
- Centered white text
- Used for: title, section dividers, summary

**Section title slides** (`<!-- _class: section-title -->`):
- Dark blue gradient
- Large centered text
- Used for: major chapter transitions

**Content slides** (default):
- White background with blue left border
- Page numbers top-right
- Footer with KTH branding

**Iframe slides** (`<!-- _class: iframe-slide -->`):
- Reduced padding for maximum iframe space
- Used for: Sankey diagram embeds

## Interactive Elements

### Sankey Diagrams

Two interactive Sankey diagrams show command usage evolution:

1. **Early Training** (`early_training_sankey_T25_2m8geyey.html`):
   - Command patterns at training start
   - Shows initial exploration behavior

2. **Late Training** (`late_training_sankey_T25_2m8geyey.html`):
   - Command patterns after convergence
   - Demonstrates learned efficiency

**Viewing**: Diagrams are fully interactive in HTML presentation mode (hover, zoom, pan)

## Mathematics

Math rendering uses KaTeX (enabled via `math: katex` in frontmatter):

- **Inline**: `\( expression \)` - e.g., \( \pi_\theta(y | x) \)
- **Display**: `\[ expression \]` - e.g., equations on dedicated lines

**Examples from slides**:
- PPO clipped surrogate objective
- GRPO group-relative advantage
- GSPO sequence-level importance ratio
- Patch-similarity reward aggregation

## Code Highlighting

Code blocks use Marp's built-in syntax highlighting:

```python
>>> shell(cmd="grep -n 'def process' src/utils.py")
42:def process_data(data):
43:    return data.strip().lower()
```

Styling matches thesis minted configuration (light background, blue left border).

## Asset Organization

```
slides/
├── presentation.md          # Main slide deck (40 slides)
├── kth-theme.css           # Custom KTH styling
├── images/                 # Copied assets
│   ├── kth-logo.jpg       # KTH logo (top-right on slides)
│   └── kth-footer.png     # KTH footer image
└── README.md              # This file
```

### External References

Slides reference figures from parent directories:

- `../plotting/figures/nano_blank.png` - Nano agent icon
- `../plotting/figures/training_sequence_diagram.png` - Training architecture
- `../plotting/figures/plots/sankey/*.html` - Interactive Sankey diagrams

**Important**: When distributing slides, ensure these relative paths remain valid or copy assets locally.

## Presentation Tips

### Timing Guidance

| Section | Slides | Time | Notes |
|---------|--------|------|-------|
| Introduction | 1-10 | 2-3 min | Quick motivation, RQs |
| Background | 11-20 | 5-6 min | Emphasize RL progression |
| Methodology | 21-28 | 4-5 min | Show Nano simplicity |
| Implementation | 29-38 | 5-6 min | Highlight innovations |
| Buffer | - | 2-3 min | Questions, flexibility |

### Key Slides to Emphasize

1. **Slide 4** (Research Questions): Core of thesis
2. **Slide 12** (PPO → GRPO → GSPO): Mathematical progression
3. **Slide 20** (Related Work): Positioning vs. DeepSWE (80× less compute)
4. **Slide 24** (Tool Example): Nano simplicity demo
5. **Slides 37-38** (Sankey): Visual training dynamics

### Presentation Mode

**Browser presentation** (recommended):
1. Open `presentation.html` in Chrome/Firefox
2. Press `F11` for fullscreen
3. Use arrow keys or click to navigate
4. Sankey diagrams will be fully interactive

**VS Code presentation**:
1. Open `presentation.md` in VS Code with Marp extension
2. Use Marp preview pane
3. Present directly from editor

## Customization

### Adjusting Content

Edit `presentation.md` to:
- Add/remove slides (maintain `---` separators)
- Update mathematical notation
- Adjust code examples
- Modify text content

### Styling Changes

Edit `kth-theme.css` to:
- Adjust colors (change CSS custom properties in `:root`)
- Modify typography (font families, sizes)
- Update layout (padding, margins, grid columns)
- Customize slide classes

### Adding Images

1. Copy image to `images/` directory
2. Reference in markdown: `![alt text](images/filename.png)`
3. Optional sizing: `![width:600px](images/filename.png)`

## Exporting for Distribution

### Standalone HTML Package

```bash
# Build with base64-encoded assets (self-contained)
marp presentation.md --html --theme kth-theme.css --allow-local-files -o presentation-standalone.html
```

### PDF for Printing

```bash
# High-quality PDF (note: iframes won't render)
marp presentation.md --theme kth-theme.css --pdf-notes --allow-local-files -o presentation-print.pdf
```

### Archive for Submission

```bash
cd ..
zip -r thesis-slides.zip slides/ -x "slides/.DS_Store" "slides/*.html" "slides/*.pdf"
```

## Troubleshooting

### Iframes Not Loading

**Issue**: Sankey diagrams don't appear
**Solution**: Ensure `--html` flag is used and paths are correct

```bash
marp presentation.md --html --theme kth-theme.css -o presentation.html
```

### Math Not Rendering

**Issue**: LaTeX appears as raw text
**Solution**: Check frontmatter has `math: katex`

### Images Missing

**Issue**: Figures don't display
**Solution**: Verify relative paths from slides directory:
```bash
ls ../plotting/figures/nano_blank.png
ls ../plotting/figures/plots/sankey/early_training_sankey_T25_2m8geyey.html
```

### Theme Not Applied

**Issue**: Default Marp styling appears
**Solution**: Specify theme explicitly:
```bash
marp presentation.md --theme kth-theme.css -o presentation.html
```

Or ensure `theme: kth-theme` in frontmatter

### Browser Compatibility

**Recommended**: Chrome or Firefox (latest versions)
**Works**: Safari, Edge
**Issues**: Older browsers may have CSS grid or KaTeX problems

## Development Workflow

1. **Edit** slides in `presentation.md`
2. **Preview** with VS Code Marp extension or `marp -s -w`
3. **Test** build with `marp --html --theme kth-theme.css presentation.md`
4. **Iterate** based on timing and content flow
5. **Finalize** with full build including PDF backup

## Resources

- [Marp Documentation](https://marpit.marp.app/)
- [Marp CLI](https://github.com/marp-team/marp-cli)
- [VS Code Extension](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)
- [KaTeX Math Functions](https://katex.org/docs/supported.html)
- [Markdown Syntax](https://www.markdownguide.org/cheat-sheet/)

## License & Attribution

These slides are part of a KTH master's thesis:
- **Author**: Bjarni Haukur Bjarnason
- **Institution**: KTH Royal Institute of Technology
- **Year**: 2025

KTH branding used in accordance with institutional guidelines.

