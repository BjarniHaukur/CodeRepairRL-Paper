#!/usr/bin/env python3
"""
Training Reward Comparison: Qwen3-14B vs Llama3.1-8B

Compares training reward progression between two GRPO-trained models:
- Qwen3-14B-Multilingual-GSPO (6wkkt1s0): High-performing larger model
- Llama3.1-8B-Multilingual-GSPO-Nano (1doecift): Lower-performing smaller model

This plot demonstrates the dramatic 33x performance gap between the models,
with Qwen3-14B reaching 0.2884 reward vs Llama3.1-8B reaching only 0.0087.

Key finding: Model size significantly impacts RL training effectiveness.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from wandb_utils import get_run, get_history
from plot_config import (
    create_figure, save_figure, format_axis_labels,
    get_color, FONT_CONFIG
)


# Configuration
ENTITY = "assert-kth"
PROJECT = "SWE-Gym-GRPO"

# Run IDs with URLs for reference
RUN_QWEN_14B = "6wkkt1s0"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/6wkkt1s0
RUN_LLAMA_8B = "1doecift"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/1doecift

# Plot settings
EMA_ALPHA = 0.01  # Smoothing factor for exponential moving average
RAW_ALPHA = 0.2   # Transparency for raw data
SMOOTHED_LINEWIDTH = 2.5
RAW_LINEWIDTH = 1.0


def exponential_moving_average(data, alpha=0.01):
    """
    Apply exponential moving average smoothing to data.

    Args:
        data: Array-like data to smooth
        alpha: Smoothing factor (lower = smoother)

    Returns:
        Smoothed data array
    """
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]

    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]

    return smoothed


def create_stats_box(ax, run1_final, run2_final, run1_name, run2_name):
    """
    Add a statistics box comparing final rewards.

    Args:
        ax: Matplotlib axes
        run1_final: Final reward for run 1
        run2_final: Final reward for run 2
        run1_name: Name of run 1
        run2_name: Name of run 2
    """
    # Calculate performance ratio
    ratio = run1_final / run2_final if run2_final > 0 else float('inf')

    # Create stats text
    stats_text = "Final Rewards:\n"
    stats_text += f"{run1_name}: {run1_final:.4f}\n"
    stats_text += f"{run2_name}: {run2_final:.4f}\n"
    stats_text += f"Ratio: {ratio:.1f}x"

    # Position in upper left
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5',
                     facecolor='white',
                     edgecolor='gray',
                     alpha=0.9),
            fontsize=FONT_CONFIG["size"]["small"],
            family='monospace')


def main():
    """Create reward comparison plot between Qwen3-14B and Llama3.1-8B."""

    parser = argparse.ArgumentParser(description='Compare training rewards between two runs')
    parser.add_argument('--max-y', type=float, default=None,
                        help='Maximum y-axis value (default: auto)')
    parser.add_argument('--show-stats', action='store_true', default=False,
                        help='Show statistics box in top-left corner')
    args = parser.parse_args()

    print("="*70)
    print("Training Reward Comparison: Qwen3-14B vs Llama3.1-8B")
    print("="*70)

    # Load runs
    print("\n1. Loading runs...")
    run_qwen = get_run(ENTITY, PROJECT, RUN_QWEN_14B)
    run_llama = get_run(ENTITY, PROJECT, RUN_LLAMA_8B)

    # Get training reward history
    print("\n2. Fetching training rewards...")
    history_qwen = get_history(run_qwen, keys=["train/reward", "train/global_step"])
    history_llama = get_history(run_llama, keys=["train/reward", "train/global_step"])

    # Clean data - remove NaN values
    print("\n3. Cleaning data...")
    mask_qwen = ~history_qwen["train/reward"].isna()
    mask_llama = ~history_llama["train/reward"].isna()

    steps_qwen = history_qwen.loc[mask_qwen, "train/global_step"].values
    rewards_qwen = history_qwen.loc[mask_qwen, "train/reward"].values

    steps_llama = history_llama.loc[mask_llama, "train/global_step"].values
    rewards_llama = history_llama.loc[mask_llama, "train/reward"].values

    print(f"Qwen3-14B: {len(rewards_qwen)} data points, final reward = {rewards_qwen[-1]:.4f}")
    print(f"Llama3.1-8B: {len(rewards_llama)} data points, final reward = {rewards_llama[-1]:.4f}")

    # Apply EMA smoothing
    print(f"\n4. Applying EMA smoothing (alpha={EMA_ALPHA})...")
    smoothed_qwen = exponential_moving_average(rewards_qwen, alpha=EMA_ALPHA)
    smoothed_llama = exponential_moving_average(rewards_llama, alpha=EMA_ALPHA)

    # Create figure
    print("\n5. Creating publication-ready plot...")
    fig, ax = create_figure(size="large")

    # Define colors
    color_qwen = get_color("primary")    # Blue for Qwen3-14B
    color_llama = get_color("secondary")  # Red for Llama3.1-8B

    # Plot Qwen3-14B (better performer)
    # Raw data (transparent)
    ax.plot(steps_qwen, rewards_qwen,
            color=color_qwen,
            alpha=RAW_ALPHA,
            linewidth=RAW_LINEWIDTH,
            label='_nolegend_')  # Hide from legend

    # Smoothed data
    ax.plot(steps_qwen, smoothed_qwen,
            color=color_qwen,
            linewidth=SMOOTHED_LINEWIDTH,
            label='Qwen3-14B-Multilingual-GSPO',
            zorder=10)  # Draw on top

    # Plot Llama3.1-8B (lower performer)
    # Raw data (transparent)
    ax.plot(steps_llama, rewards_llama,
            color=color_llama,
            alpha=RAW_ALPHA,
            linewidth=RAW_LINEWIDTH,
            label='_nolegend_')  # Hide from legend

    # Smoothed data
    ax.plot(steps_llama, smoothed_llama,
            color=color_llama,
            linewidth=SMOOTHED_LINEWIDTH,
            label='Llama3.1-8B-Multilingual-GSPO-Nano',
            zorder=9)

    # Format axes
    format_axis_labels(
        ax,
        xlabel="Training Steps",
        ylabel="Reward",
        title="Training Reward Comparison: Model Size Impact on RL Performance"
    )

    # Add legend (forced to top right)
    ax.legend(loc='upper right',
             frameon=True,
             framealpha=0.9,
             edgecolor='gray',
             fontsize=FONT_CONFIG["size"]["medium"])

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add stats box (optional, controlled by --show-stats)
    if args.show_stats:
        create_stats_box(ax,
                        rewards_qwen[-1],
                        rewards_llama[-1],
                        "Qwen3-14B",
                        "Llama3.1-8B")

    # Set y-axis limits
    if args.max_y is not None:
        ax.set_ylim(0, args.max_y)
        print(f"\nUsing manual y-axis limit: 0 to {args.max_y}")
    else:
        # Ensure y-axis starts at 0 to show full scale of difference
        ax.set_ylim(bottom=0)
        print("\nUsing automatic y-axis scaling (starting from 0)")

    # Format y-axis to show more precision
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

    # Save figure with both run IDs in filename
    print("\n6. Saving figure...")
    save_figure(fig,
               "reward_comparison_6wkkt1s0_vs_1doecift",
               plot_type="comparison")

    # Print key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print(f"Qwen3-14B (larger model):")
    print(f"  - Initial reward: {rewards_qwen[0]:.4f}")
    print(f"  - Final reward:   {rewards_qwen[-1]:.4f}")
    print(f"  - Improvement:    {rewards_qwen[-1] - rewards_qwen[0]:.4f}")
    print(f"  - Training steps: {len(rewards_qwen)} data points")
    print()
    print(f"Llama3.1-8B (smaller model):")
    print(f"  - Initial reward: {rewards_llama[0]:.4f}")
    print(f"  - Final reward:   {rewards_llama[-1]:.4f}")
    print(f"  - Improvement:    {rewards_llama[-1] - rewards_llama[0]:.4f}")
    print(f"  - Training steps: {len(rewards_llama)} data points")
    print()

    if rewards_llama[-1] > 0:
        ratio = rewards_qwen[-1] / rewards_llama[-1]
        print(f"Performance Gap:")
        print(f"  - Qwen3-14B achieves {ratio:.1f}x higher final reward")
        print(f"  - Absolute difference: {rewards_qwen[-1] - rewards_llama[-1]:.4f}")

    print("\nConclusion:")
    print("  Model size has a dramatic impact on RL training effectiveness.")
    print("  The 14B model significantly outperforms the 8B model in learning")
    print("  from reinforcement signals, suggesting larger models better capture")
    print("  the complex patterns needed for code repair tasks.")
    print("="*70)

    plt.close()


if __name__ == "__main__":
    main()
