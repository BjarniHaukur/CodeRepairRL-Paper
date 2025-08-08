# CodeRepairRL Plotting Framework

A modular, LLM-friendly plotting framework for visualizing thesis results from Weights & Biases experiments.

## Setup

```bash
cd plotting
pip install -r requirements.txt
```

Set your W&B credentials:
```bash
export WANDB_API_KEY="your-api-key"
# Or create a .env file with:
# WANDB_API_KEY=your-api-key
```

## Structure

```
plotting/
├── wandb_utils.py           # Core W&B API utilities
├── plot_config.py           # Shared styling and configuration
├── plots/                   # Individual plot scripts
│   ├── loss_curves.py       # Training loss over time
│   ├── success_rate_comparison.py  # Bar charts and model scaling
│   └── ...                  # Additional plots
└── README.md               # This file
```

## Usage

### Basic Example

```python
from plots.loss_curves import plot_loss_curves

plot_loss_curves(
    entity="your-wandb-entity",
    project="coderepair-rl",
    tags_nano=["nano-agent", "qwen3-7b"],
    tags_heavy=["heavyweight", "qwen3-7b"],
    metric="train/loss",
    output_name="training_loss"
)
```

### Available Plots

1. **Loss Curves** (`loss_curves.py`)
   - Compares training loss between Nano agent and heavyweight scaffolding
   - Shows mean and standard deviation across seeds
   - Outputs: `figures/plots/loss_curves.png`

2. **Success Rate Comparison** (`success_rate_comparison.py`)
   - Bar chart comparing repair success rates
   - Model size scaling analysis
   - Outputs: `figures/plots/success_rate_comparison.png`

### Creating New Plots

1. Create a new file in `plots/` directory
2. Import utilities:
   ```python
   import sys
   import os
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   
   from wandb_utils import fetch_runs, get_metrics
   from plot_config import create_figure, save_figure, get_color
   ```

3. Follow the pattern of existing plots:
   - Fetch runs using tags
   - Extract metrics with verbose output
   - Create figure with consistent styling
   - Save to `figures/plots/`

## LLM-Friendly Features

- **Verbose Output**: All functions print detailed information about data being processed
- **Data Inspection**: `print_summary()` shows keys, means, standard deviations
- **Clear Structure**: Each plot is self-contained with descriptive variable names
- **Consistent Naming**: Output files match script names for easy tracking

## Common W&B Tags

Based on your research:
- Scaffold types: `"nano-agent"`, `"heavyweight"`, `"aider"`
- Model sizes: `"qwen3-1b"`, `"qwen3-7b"`, `"qwen3-14b"`
- Training stages: `"sft"`, `"rl"`, `"grpo"`
- Datasets: `"swe-gym"`, `"swe-bench-verified"`

## Troubleshooting

- **No runs found**: Check entity/project names and tags
- **Missing metrics**: Verify metric names match W&B logging
- **Import errors**: Ensure you're running from the `plotting/` directory
- **API errors**: Check WANDB_API_KEY is set correctly

## Extending the Framework

### Adding New Utility Functions

Edit `wandb_utils.py` to add functions for:
- Different aggregation methods
- Custom filtering logic
- Specialized data transformations

### Customizing Plot Styles

Edit `plot_config.py` to modify:
- Color schemes (COLORS dict)
- Figure sizes (FIGURE_SIZES dict)
- Font settings (FONT_CONFIG dict)
- Export settings (EXPORT_CONFIG dict)