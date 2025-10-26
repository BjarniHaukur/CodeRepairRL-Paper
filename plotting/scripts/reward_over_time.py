#!/usr/bin/env python3
"""
Plot reward mean and standard deviation over training steps.
Shows both train/reward (mean) and train/reward_std in subplots.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from wandb_utils import get_run, get_history
from plot_config import ENTITY, PROJECT, RUN_ID, get_output_filename, setup_plotting_style

def apply_ema_smoothing(values, alpha=0.05):
    """Apply exponential moving average smoothing."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]

    # Skip first 0.5% to avoid startup bias
    skip_points = max(1, int(len(smoothed) * 0.005))
    return smoothed[skip_points:], skip_points

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Plot reward mean and std over time')
    parser.add_argument('--run-id', type=str, default=RUN_ID,
                        help=f'WandB run ID (default: {RUN_ID})')
    parser.add_argument('--merge-with', type=str, default=None,
                        help='Optional second run ID to merge with (for continued training runs)')
    parser.add_argument('--ema-alpha', type=float, default=0.05,
                        help='EMA smoothing parameter (default: 0.05)')
    args = parser.parse_args()

    print("="*60)
    print("Reward Over Training")
    print("="*60)

    # Get run and history (with optional merging)
    if args.merge_with:
        print(f"\n🔗 Merging runs: {args.run_id} + {args.merge_with}")
        from scripts.merge_runs import merge_continued_runs
        merged_history, run, _ = merge_continued_runs(args.run_id, args.merge_with, ENTITY, PROJECT)
        history = merged_history[['_step', 'train/reward', 'train/reward_std']]
    else:
        run = get_run(ENTITY, PROJECT, args.run_id)
        metrics = ['_step', 'train/reward', 'train/reward_std']
        history = get_history(run, keys=metrics)

    # Filter out NaN values
    history = history.dropna(subset=['train/reward', 'train/reward_std'])
    
    if history.empty:
        print("❌ No reward data found")
        return

    print(f"Found {len(history)} reward evaluations")

    # Set up the plot with side-by-side subplots
    setup_plotting_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Get data
    reward_mean = history['train/reward'].values
    reward_std = history['train/reward_std'].values
    training_steps = np.arange(len(history))  # Use sequential index as training steps

    # Apply EMA smoothing to both
    mean_smoothed, skip_points = apply_ema_smoothing(reward_mean, args.ema_alpha)
    std_smoothed, _ = apply_ema_smoothing(reward_std, args.ema_alpha)
    steps_smoothed = training_steps[skip_points:]

    # Plot reward mean (left panel)
    ax1.plot(training_steps, reward_mean,
            color='#3498DB', linewidth=1, alpha=0.2, zorder=1)
    ax1.plot(steps_smoothed, mean_smoothed,
            color='#3498DB', linewidth=2.5, alpha=0.9, zorder=2)
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Reward', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Set y-axis based on EMA values (starting from 0)
    mean_max = mean_smoothed.max()
    mean_margin = mean_max * 0.1
    ax1.set_ylim(0, mean_max + mean_margin)

    # Plot reward std (right panel)
    ax2.plot(training_steps, reward_std,
            color='#3498DB', linewidth=1, alpha=0.2, zorder=1)
    ax2.plot(steps_smoothed, std_smoothed,
            color='#3498DB', linewidth=2.5, alpha=0.9, zorder=2)
    ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reward Std Dev', fontsize=12, fontweight='bold')
    ax2.set_title('Reward Standard Deviation', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Set y-axis based on EMA values (starting from 0)
    std_max = std_smoothed.max()
    std_margin = std_max * 0.1
    ax2.set_ylim(0, std_max + std_margin)

    # Print statistics
    print(f"\nReward statistics:")
    print(f"  Mean - Mean: {mean_smoothed.mean():.4f}, Final: {mean_smoothed[-1]:.4f}")
    print(f"  Std - Mean: {std_smoothed.mean():.4f}, Final: {std_smoothed[-1]:.4f}")

    # Adjust layout and save
    plt.tight_layout()

    output_path = get_output_filename(f"reward_over_time", args.run_id, plot_type="temporal")
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}.png")

    print(f"\n✅ Reward plot created successfully!")

if __name__ == "__main__":
    main()