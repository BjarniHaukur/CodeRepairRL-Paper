#!/usr/bin/env python3
"""
Compare tool success rates across multiple model runs.
Side-by-side subplots for shell and apply_patch success rates.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from wandb_utils import get_run, get_history
from plot_config import ENTITY, PROJECT, get_output_filename, setup_plotting_style

# Hardcoded model configurations (same as reward_comparison.py)
MODELS = {
    'yhj2yemg': {'name': 'Qwen3-32B', 'color': '#E74C3C', 'max_steps': 135},  # Red - cut at 135
    'ajgo643t': {'name': 'Qwen3-8B', 'color': '#F39C12', 'max_steps': None},   # Orange
    'jb5uxlqc': {'name': 'Qwen3-14B', 'color': '#8E44AD', 'max_steps': None},  # Purple - FIXED
    '1doecift': {'name': 'Llama3.1-8B', 'color': '#27AE60', 'max_steps': None} # Green
}

def apply_rolling_average(data, window=10):
    """
    Apply rolling window average smoothing.
    This preserves the actual scale better than EMA.
    """
    import pandas as pd

    # Use pandas rolling average (centered for better smoothing)
    smoothed = pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values

    return smoothed

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare tool success rates across multiple models')
    parser.add_argument('--window', type=int, default=10,
                        help='Rolling average window size (default: 10)')
    args = parser.parse_args()

    print("="*60)
    print("Multi-Model Tool Success Rate Comparison")
    print("="*60)

    # Set up the plot with subplots
    setup_plotting_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Track max steps for x-axis limit
    max_training_step = 0

    # Process each model
    for run_id, config in MODELS.items():
        print(f"\nProcessing {config['name']} (run: {run_id})...")

        # Get run and history
        run = get_run(ENTITY, PROJECT, run_id)
        history = get_history(run, keys=['_step',
                                        'train/extra_kwargs/tool_success_rate_shell',
                                        'train/extra_kwargs/tool_success_rate_apply_patch'])

        # Filter rows with required metrics
        history = history.dropna(subset=[
            'train/extra_kwargs/tool_success_rate_shell',
            'train/extra_kwargs/tool_success_rate_apply_patch'
        ])

        if history.empty:
            print(f"  ❌ No success rate data found for {config['name']}")
            continue

        # Apply step limit if specified
        if config['max_steps'] is not None:
            history = history[history.index <= config['max_steps']]
            print(f"  Limiting to {config['max_steps']} steps")

        # Get success rates (already calculated in wandb)
        shell_rate = history['train/extra_kwargs/tool_success_rate_shell'].values
        patch_rate = history['train/extra_kwargs/tool_success_rate_apply_patch'].values

        print(f"  Found {len(shell_rate)} data points")

        # Apply rolling average smoothing
        shell_smoothed = apply_rolling_average(shell_rate, window=args.window)
        patch_smoothed = apply_rolling_average(patch_rate, window=args.window)
        training_steps = np.arange(len(shell_smoothed))

        # Track maximum step for axis limits
        if len(training_steps) > 0:
            max_training_step = max(max_training_step, training_steps[-1])

        # Plot shell success rate on left subplot
        ax1.plot(training_steps, shell_smoothed * 100,
                label=config['name'], color=config['color'],
                linewidth=2.5, alpha=0.9)

        # Plot apply_patch success rate on right subplot
        ax2.plot(training_steps, patch_smoothed * 100,
                label=config['name'], color=config['color'],
                linewidth=2.5, alpha=0.9)

        print(f"  Shell success: {shell_smoothed.mean()*100:.1f}%, Final: {shell_smoothed[-1]*100:.1f}%")
        print(f"  Apply patch success: {patch_smoothed.mean()*100:.1f}%, Final: {patch_smoothed[-1]*100:.1f}%")

    # Styling for shell subplot (left)
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Shell Command Success Rate', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, loc='best')
    ax1.set_ylim(0, 100)

    # Remove duplicate "0" and set x-axis limits for shell subplot
    xticks1 = ax1.get_xticks()
    xticks1 = xticks1[xticks1 > 0]
    ax1.set_xticks(xticks1)
    ax1.set_xlim(0, max_training_step)

    # Styling for apply_patch subplot (right)
    ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Apply Patch Success Rate', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, loc='best')
    ax2.set_ylim(0, 100)

    # Remove duplicate "0" and set x-axis limits for apply_patch subplot
    xticks2 = ax2.get_xticks()
    xticks2 = xticks2[xticks2 > 0]
    ax2.set_xticks(xticks2)
    ax2.set_xlim(0, max_training_step)

    # Overall title
    fig.suptitle('Model Comparison: Tool Success Rates Over Training',
                fontsize=16, fontweight='bold', y=1.02)

    # Adjust layout and save
    plt.tight_layout()

    output_path = get_output_filename(f"tool_success_comparison_win{args.window}", "multi", plot_type="comparison")
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to: {output_path}.png")

    print(f"\n✅ Tool success rate comparison plot created successfully!")

if __name__ == "__main__":
    main()
