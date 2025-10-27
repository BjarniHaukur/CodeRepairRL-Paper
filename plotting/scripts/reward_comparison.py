#!/usr/bin/env python3
"""
Compare mean reward across multiple model runs.
Single plot showing EMA-smoothed reward curves for different models.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from wandb_utils import get_run, get_history
from plot_config import ENTITY, PROJECT, get_output_filename, setup_plotting_style

# Hardcoded model configurations
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
    parser = argparse.ArgumentParser(description='Compare reward across multiple models')
    parser.add_argument('--window', type=int, default=10,
                        help='Rolling average window size (default: 10)')
    args = parser.parse_args()

    print("="*60)
    print("Multi-Model Reward Comparison")
    print("="*60)

    # Set up the plot
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Track max steps for x-axis limit
    max_training_step = 0

    # Process each model
    for run_id, config in MODELS.items():
        print(f"\nProcessing {config['name']} (run: {run_id})...")

        # Get run and history
        run = get_run(ENTITY, PROJECT, run_id)
        history = get_history(run, keys=['_step', 'train/reward'])
        history = history.dropna(subset=['train/reward'])

        if history.empty:
            print(f"  ❌ No reward data found for {config['name']}")
            continue

        # Apply step limit if specified
        if config['max_steps'] is not None:
            history = history[history.index <= config['max_steps']]
            print(f"  Limiting to {config['max_steps']} steps")

        # Get reward data
        reward_mean = history['train/reward'].values
        print(f"  Found {len(reward_mean)} data points")

        # Apply rolling average smoothing
        mean_smoothed = apply_rolling_average(reward_mean, window=args.window)
        training_steps = np.arange(len(mean_smoothed))

        # Track maximum step for axis limits
        if len(training_steps) > 0:
            max_training_step = max(max_training_step, training_steps[-1])

        # Plot smoothed reward (no raw data)
        ax.plot(training_steps, mean_smoothed,
               label=config['name'], color=config['color'],
               linewidth=2.5, alpha=0.9)

        print(f"  Mean reward: {mean_smoothed.mean():.4f}, Final: {mean_smoothed[-1]:.4f}")

    # Styling
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=14, fontweight='bold')
    ax.set_title('Model Comparison: Mean Reward Over Training',
                fontsize=16, fontweight='bold', pad=20)

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='best')

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Remove duplicate "0" label on x-axis
    xticks = ax.get_xticks()
    xticks = xticks[xticks > 0]
    ax.set_xticks(xticks)

    # Set x-axis to actual data range
    ax.set_xlim(0, max_training_step)

    # Adjust layout and save
    plt.tight_layout()

    output_path = get_output_filename(f"reward_comparison_win{args.window}", "multi", plot_type="comparison")
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to: {output_path}.png")

    print(f"\n✅ Reward comparison plot created successfully!")

if __name__ == "__main__":
    main()
