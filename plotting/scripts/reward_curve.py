#!/usr/bin/env python3
"""
Plot training reward over training steps with EMA smoothing.

This is THE primary RL learning signal showing how the agent's performance
improves over the course of training.

Run ID: 6wkkt1s0 (https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/6wkkt1s0)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from wandb_utils import get_run, get_history
from plot_config import create_figure, save_figure, format_axis_labels, get_color


# Configuration
ENTITY = "assert-kth"
PROJECT = "SWE-Gym-GRPO"
RUN_ID = "6wkkt1s0"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/6wkkt1s0


def apply_ema_smoothing(values, alpha=0.05):
    """
    Apply exponential moving average smoothing.

    Args:
        values: Array of values to smooth
        alpha: Smoothing parameter (0 < alpha < 1, lower = more smoothing)

    Returns:
        Tuple of (smoothed array with first 0.5% removed, skip_points)
    """
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]

    # Skip first 0.5% to avoid startup bias
    skip_points = max(1, int(len(smoothed) * 0.005))
    return smoothed[skip_points:], skip_points


def main():
    """Plot reward curve over training with EMA smoothing."""
    parser = argparse.ArgumentParser(description='Plot training reward over time')
    parser.add_argument('--run-id', type=str, default=RUN_ID,
                        help=f'WandB run ID (default: {RUN_ID})')
    parser.add_argument('--ema-alpha', type=float, default=0.05,
                        help='EMA smoothing parameter (default: 0.05)')
    args = parser.parse_args()

    print("="*60)
    print("Training Reward Curve")
    print("="*60)

    # Get run
    print(f"\n1. Loading run {args.run_id}...")
    run = get_run(ENTITY, PROJECT, args.run_id)
    print(f"   Run name: {run.name}")

    # Get reward metric
    print(f"\n2. Fetching training reward data...")
    metrics = ['train/global_step', 'train/reward']
    history = get_history(run, keys=metrics)

    # Filter out NaN values
    history = history.dropna(subset=['train/reward'])

    if history.empty:
        print("❌ No reward data found")
        return

    print(f"   Found {len(history)} data points")

    # Extract data
    steps = history['train/global_step'].values
    reward_values = history['train/reward'].values

    print(f"\n3. Computing statistics...")
    # Apply EMA smoothing
    reward_smoothed, skip_points = apply_ema_smoothing(reward_values, alpha=args.ema_alpha)
    steps_smoothed = steps[skip_points:]

    # Calculate statistics
    initial_reward = reward_smoothed[0]
    final_reward = reward_smoothed[-1]
    mean_reward = reward_smoothed.mean()
    improvement = ((final_reward - initial_reward) / abs(initial_reward)) * 100 if initial_reward != 0 else float('inf')

    print(f"   Initial reward: {initial_reward:.4f}")
    print(f"   Final reward: {final_reward:.4f}")
    print(f"   Mean reward: {mean_reward:.4f}")
    print(f"   Improvement: {improvement:.1f}%")

    # Create plot
    print(f"\n4. Creating plot...")
    fig, ax = create_figure(size="large")

    # Plot raw data with transparency
    ax.plot(steps, reward_values,
            color=get_color("primary"),
            linewidth=1,
            alpha=0.3,
            label='Raw Data')

    # Plot EMA smoothed data
    ax.plot(steps_smoothed, reward_smoothed,
            color=get_color("primary"),
            linewidth=2.5,
            alpha=0.9,
            label=f'EMA Smoothed (α={args.ema_alpha})')

    # Format axes
    format_axis_labels(ax,
                       xlabel='Training Steps',
                       ylabel='Reward',
                       title=f'{run.name}: Training Reward Curve')

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)

    # Set y-axis limits with margin (use all data for range calculation)
    y_min, y_max = reward_values.min(), reward_values.max()
    margin = (y_max - y_min) * 0.1  # 10% margin
    ax.set_ylim(max(0, y_min - margin), y_max + margin)

    # Add statistics box
    stats_text = (f'Initial: {initial_reward:.4f}\n'
                  f'Final: {final_reward:.4f}\n'
                  f'Mean: {mean_reward:.4f}\n'
                  f'Improvement: {improvement:.1f}%')

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            fontsize=10)

    # Save figure
    print(f"\n5. Saving figure...")
    output_filename = f"reward_curve_ema{args.ema_alpha}_{args.run_id}"
    save_figure(fig, output_filename, plot_type="temporal")

    print(f"\n✅ Reward curve plot created successfully!")
    print(f"   Output: figures/plots/temporal/{output_filename}.png")

    plt.close()


if __name__ == "__main__":
    main()
