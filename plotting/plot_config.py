"""
Common configuration for all plots in the CodeRepairRL thesis.
Provides consistent styling, colors, and export settings.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional

# WandB configuration constants
ENTITY = "assert-kth"
PROJECT = "SWE-Gym-GRPO"
RUN_ID = "nz1r7ml3"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/nz1r7ml3

# Standardized filename format with plot type organization
FILENAME_FORMAT = "{name}_{run_id}"

# Plot type categories for organized file structure
PLOT_TYPES = {
    "analysis": ["distribution", "clustering", "performance", "problem"],
    "temporal": ["evolution", "timeline", "training", "reward"],
    "sankey": ["command", "flow", "transition"],
    "comparison": ["vs", "compare", "baseline"]
}

def get_output_filename(name: str, run_id: str = None, plot_type: str = None) -> str:
    """
    Generate standardized output filename with optional plot type categorization.
    
    Args:
        name: Base name for the plot
        run_id: WandB run ID (defaults to global RUN_ID)
        plot_type: Plot category (analysis, temporal, sankey, comparison)
        
    Returns:
        Full path for saving the plot
    """
    if run_id is None:
        run_id = RUN_ID
    
    filename = FILENAME_FORMAT.format(name=name, run_id=run_id)
    
    # Auto-detect plot type if not specified
    if plot_type is None:
        plot_type = _auto_detect_plot_type(name)
    
    return f"figures/plots/{plot_type}/{filename}"

def _auto_detect_plot_type(name: str) -> str:
    """Auto-detect plot type based on name keywords."""
    name_lower = name.lower()
    
    for category, keywords in PLOT_TYPES.items():
        if any(keyword in name_lower for keyword in keywords):
            return category
    
    # Default to analysis if no match
    return "analysis"

# Color scheme for different components
COLORS = {
    # Scaffold types
    "nano-agent": "#2E86C1",      # Blue
    "nano_agent": "#2E86C1",      # Blue (alternative naming)
    "heavyweight": "#E74C3C",     # Red
    "aider": "#E74C3C",          # Red (specific heavyweight)
    "baseline": "#95A5A6",        # Gray
    
    # Model sizes
    "qwen3-1b": "#F39C12",       # Orange
    "qwen3-7b": "#27AE60",       # Green
    "qwen3-14b": "#8E44AD",      # Purple
    
    # Training stages
    "sft": "#3498DB",            # Light blue
    "rl": "#E67E22",             # Dark orange
    "grpo": "#E67E22",           # Same as RL
    
    # General purpose
    "primary": "#2E86C1",
    "secondary": "#E74C3C",
    "tertiary": "#27AE60",
    "quaternary": "#F39C12",
    "success": "#27AE60",
    "error": "#E74C3C",
    "neutral": "#95A5A6"
}

# Command category colors for Sankey diagrams
COMMAND_COLORS = {
    # Critical tool - dominates red channel
    "apply_patch": "#DC143C",  # Crimson red - most important, stands out
    
    # Exploration commands (blues - keep rg/grep similar but distinguishable)
    "rg": "#3498DB",           # Bright blue
    "grep": "#2E86C1",         # Similar blue but darker
    "find": "#1ABC9C",         # Distinct teal
    "ls": "#16A085",           # Dark teal
    
    # File reading (greens - more variation)
    "cat": "#27AE60",          # Green
    "head": "#2ECC71",         # Light green
    "tail": "#58D68D",         # Pale green
    
    # Development tools (purples - pip and python together)
    "python": "#8E44AD",       # Purple
    "python3": "#A569BD",      # Light purple (more distinct)
    "pip": "#7B68EE",          # Medium slate blue (clearly purple)
    "git": "#5B2C6F",          # Dark purple (darker)
    
    # Text processing (oranges/yellows - avoid red channel)
    "sed": "#FF8C00",          # Dark orange (no red conflict)
    "awk": "#FFA500",          # Orange
    "diff": "#FFD700",         # Gold (bright, distinct)
    "echo": "#F0E68C",         # Khaki (pale yellow)
    
    # File operations (browns - avoid red)
    "mkdir": "#8B4513",        # Saddle brown
    "mv": "#CD853F",           # Peru brown
    "cp": "#D2691E",           # Chocolate brown
    "touch": "#DEB887",        # Burlywood
    "chmod": "#D2B48C",        # Tan
    "rm": "#A0522D",           # Sienna (dangerous but not red like apply_patch)
    
    # System/other (grays/blues - better distinction)
    "which": "#5D6D7E",        # Dark gray
    "pwd": "#85929E",          # Blue-gray
    "cd": "#AEB6BF",           # Light blue-gray
    "curl": "#2C3E50",         # Dark blue-gray
    "wget": "#34495E",         # Darker blue-gray
    "uname": "#7B7D7D",        # Medium gray
    
    # Package managers (earth tones - avoid orange/red)
    "conda": "#556B2F",        # Dark olive green
    "apt": "#6B8E23",          # Olive drab
    "brew": "#8FBC8F",         # Dark sea green
    
    # Data tools (teals/cyans)
    "dvc": "#138D75",          # Dark teal
    "dask": "#17A2B8",         # Cyan
    "mlflow": "#20B2AA",       # Light sea green
}

# Consistent figure sizes
FIGURE_SIZES = {
    "small": (6, 4),
    "medium": (8, 6),
    "large": (10, 8),
    "wide": (12, 6),
    "square": (8, 8)
}

# Export settings
EXPORT_CONFIG = {
    "dpi": 300,
    "format": "png",
    "bbox_inches": "tight",
    "facecolor": "white",
    "edgecolor": "none"
}

# Font settings for thesis
FONT_CONFIG = {
    "family": "serif",
    "size": {
        "small": 10,
        "medium": 12,
        "large": 14,
        "title": 16
    }
}


def setup_plotting_style():
    """
    Configure matplotlib and seaborn for consistent thesis styling.
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Configure matplotlib
    plt.rcParams.update({
        # Font settings
        "font.family": FONT_CONFIG["family"],
        "font.size": FONT_CONFIG["size"]["medium"],
        
        # Figure settings
        "figure.facecolor": "white",
        "figure.edgecolor": "none",
        "figure.dpi": 100,
        
        # Axes settings
        "axes.labelsize": FONT_CONFIG["size"]["medium"],
        "axes.titlesize": FONT_CONFIG["size"]["large"],
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        
        # Legend settings
        "legend.fontsize": FONT_CONFIG["size"]["small"],
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        
        # Line settings
        "lines.linewidth": 2,
        "lines.markersize": 8,
        
        # Save settings
        "savefig.dpi": EXPORT_CONFIG["dpi"],
        "savefig.format": EXPORT_CONFIG["format"],
        "savefig.bbox": EXPORT_CONFIG["bbox_inches"]
    })
    
    print("Plotting style configured for CodeRepairRL thesis")


def get_color(key: str) -> str:
    """
    Get color for a given key with fallback to default.
    
    Args:
        key: Color key (e.g., "nano-agent", "heavyweight")
    
    Returns:
        Hex color code
    """
    return COLORS.get(key, COLORS["neutral"])


def get_command_color(command: str) -> str:
    """
    Get color for a shell command with semantic grouping.
    
    Args:
        command: Command name (e.g., "rg", "apply_patch")
    
    Returns:
        Hex color code
    """
    return COMMAND_COLORS.get(command, COLORS["neutral"])


def create_figure(size: str = "medium", **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure with consistent sizing.
    
    Args:
        size: Size preset name
        **kwargs: Additional arguments for plt.subplots
    
    Returns:
        Figure and axes objects
    """
    figsize = FIGURE_SIZES.get(size, FIGURE_SIZES["medium"])
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: str = None,
    plot_type: str = None,
    **kwargs
):
    """
    Save figure with consistent settings and organized directory structure.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename (without extension)
        output_dir: Output directory (if None, uses get_output_filename logic)
        plot_type: Plot type for auto-categorization
        **kwargs: Override export settings
    """
    # Use new directory structure if output_dir not specified
    if output_dir is None:
        if plot_type is None:
            plot_type = _auto_detect_plot_type(filename)
        output_dir = f"figures/plots/{plot_type}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build full path
    ext = kwargs.get("format", EXPORT_CONFIG["format"])
    filepath = os.path.join(output_dir, f"{filename}.{ext}")
    
    # Merge export config with overrides
    save_config = {**EXPORT_CONFIG, **kwargs}
    
    # Save figure
    fig.savefig(filepath, **save_config)
    print(f"Saved figure to: {filepath}")


def format_axis_labels(
    ax: plt.Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Apply consistent formatting to axis labels and title.
    
    Args:
        ax: Matplotlib axes
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_CONFIG["size"]["medium"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_CONFIG["size"]["medium"])
    if title:
        ax.set_title(title, fontsize=FONT_CONFIG["size"]["title"], pad=20)


def add_comparison_annotations(
    ax: plt.Axes,
    data1: float,
    data2: float,
    label1: str = "Nano-agent",
    label2: str = "Heavyweight",
    metric: str = "improvement"
):
    """
    Add annotations comparing two values.
    
    Args:
        ax: Matplotlib axes
        data1: First value
        data2: Second value
        label1: Label for first value
        label2: Label for second value
        metric: Metric name
    """
    diff = data1 - data2
    percent_diff = (diff / data2) * 100 if data2 != 0 else 0
    
    annotation = f"{label1} vs {label2}:\n"
    annotation += f"Δ = {diff:.2f} ({percent_diff:+.1f}%)"
    
    ax.text(0.02, 0.98, annotation,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=FONT_CONFIG["size"]["small"])


def get_line_styles() -> List[str]:
    """Get list of line styles for multiple lines."""
    return ['-', '--', '-.', ':']


def get_markers() -> List[str]:
    """Get list of markers for scatter plots."""
    return ['o', 's', '^', 'D', 'v', 'p', '*']


# Initialize style when module is imported
setup_plotting_style()