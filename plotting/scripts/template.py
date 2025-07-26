#!/usr/bin/env python3
"""
TEMPLATE: Copy this file to create new plots.

This template demonstrates best practices:
1. Hard-coded run IDs with W&B URLs
2. Clear configuration section
3. LLM-friendly data exploration
4. Consistent plotting style
"""


import matplotlib.pyplot as plt
from wandb_utils import get_run, get_history, get_summary_metrics
from plot_config import create_figure, save_figure, format_axis_labels, get_color, get_output_filename
from utils.table_parser import TableExtractor  # For table-based plots


# Configuration - ALWAYS include W&B URLs for easy access
ENTITY = "assert-kth"
PROJECT = "SWE-Gym-GRPO"

# Option 1: Individual variables with URLs
run_a = "qa9t88ng"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/qa9t88ng
run_b = "bu2fqmm0"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/bu2fqmm0
run_c = "c1mr1lgd"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/c1mr1lgd

# Option 2: List with URLs (for batch processing)
runs = [
    "qa9t88ng",  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/qa9t88ng
    "bu2fqmm0",  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/bu2fqmm0
    "c1mr1lgd",  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/c1mr1lgd
]


def main():
    """Template plot function."""
    
    print("="*60)
    print("Template Plot: Replace with your plot title")
    print("="*60)
    
    # Example 1: Compare two specific runs
    print("\n1. Loading runs...")
    run1 = get_run(ENTITY, PROJECT, run_a)
    run2 = get_run(ENTITY, PROJECT, run_b)
    
    # Example 2: Explore metrics (always do this first!)
    print("\n2. Exploring metrics...")
    history1 = get_history(run1, keys=["train/loss", "train/reward"])
    history2 = get_history(run2, keys=["train/loss", "train/reward"])
    
    # Example 3: Get summary metrics
    print("\n3. Summary metrics...")
    summary1 = get_summary_metrics(run1)
    summary2 = get_summary_metrics(run2)
    
    # Example 4: Create plot
    print("\n4. Creating plot...")
    fig, ax = create_figure(size="large")
    
    # Filter out NaN values
    mask1 = ~history1["train/loss"].isna()
    mask2 = ~history2["train/loss"].isna()
    
    # Plot
    ax.plot(history1.loc[mask1, "_step"], 
            history1.loc[mask1, "train/loss"],
            label=run1.name, 
            color=get_color("primary"),
            linewidth=2)
    
    ax.plot(history2.loc[mask2, "_step"], 
            history2.loc[mask2, "train/loss"],
            label=run2.name, 
            color=get_color("secondary"),
            linewidth=2)
    
    # Format
    format_axis_labels(ax,
                       xlabel="Training Steps",
                       ylabel="Loss", 
                       title="Template Plot: Replace with Your Title")
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save with new directory structure
    save_figure(fig, "template_comparison_plot", plot_type="comparison")
    
    # Example 5: Print insights for LLM
    print(f"\n📊 INSIGHTS:")
    print(f"Run 1 ({run1.name}): Final loss = {history1['train/loss'].dropna().iloc[-1]:.4f}")
    print(f"Run 2 ({run2.name}): Final loss = {history2['train/loss'].dropna().iloc[-1]:.4f}")
    
    plt.close()


if __name__ == "__main__":
    main()