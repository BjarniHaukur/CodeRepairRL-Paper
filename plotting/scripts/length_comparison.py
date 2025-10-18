#!/usr/bin/env python3
"""
Episode Length Comparison: Qwen3-14B vs Llama3.1-8B

Compares episode length progression between two GRPO-trained models:
- Qwen3-14B-Multilingual-GSPO (6wkkt1s0): Shows dramatic efficiency improvement (-82.6%)
- Llama3.1-8B-Multilingual-GSPO-Nano (1doecift): Shows NO efficiency improvement (+11.0%)

This plot demonstrates that only the larger model learns efficient problem-solving,
reducing episode length from 4072.8 to 708.6 tokens, while the smaller model
actually gets worse (2925.6 to 3248.2 tokens).

Key finding: Model size impacts ability to learn efficiency during RL training.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
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
EMA_ALPHA = 0.05  # Smoothing factor for exponential moving average
RAW_ALPHA = 0.2   # Transparency for raw data
SMOOTHED_LINEWIDTH = 2.5
RAW_LINEWIDTH = 1.0


def exponential_moving_average(data, alpha=0.05):
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


def create_stats_box(ax, run1_initial, run1_final, run2_initial, run2_final,
                     run1_name, run2_name):
    """
    Add a statistics box comparing episode length changes.

    Args:
        ax: Matplotlib axes
        run1_initial: Initial length for run 1
        run1_final: Final length for run 1
        run2_initial: Initial length for run 2
        run2_final: Final length for run 2
        run1_name: Name of run 1
        run2_name: Name of run 2
    """
    # Calculate percentage changes
    run1_pct = ((run1_final - run1_initial) / run1_initial * 100) if run1_initial > 0 else 0
    run2_pct = ((run2_final - run2_initial) / run2_initial * 100) if run2_initial > 0 else 0

    # Create stats text
    stats_text = "Length Change:\n"
    stats_text += f"{run1_name}:\n"
    stats_text += f"  {run1_initial:.1f} → {run1_final:.1f}\n"
    stats_text += f"  ({run1_pct:+.1f}%)\n"
    stats_text += f"{run2_name}:\n"
    stats_text += f"  {run2_initial:.1f} → {run2_final:.1f}\n"
    stats_text += f"  ({run2_pct:+.1f}%)"

    # Position in upper right
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5',
                     facecolor='white',
                     edgecolor='gray',
                     alpha=0.9),
            fontsize=FONT_CONFIG["size"]["small"],
            family='monospace')


def main():
    """Create episode length comparison plot between Qwen3-14B and Llama3.1-8B."""

    parser = argparse.ArgumentParser(description='Compare episode lengths between two runs')
    parser.add_argument('--max-y', type=float, default=None,
                        help='Maximum y-axis value (default: auto)')
    parser.add_argument('--show-stats', action='store_true', default=False,
                        help='Show statistics box in top-right corner')
    args = parser.parse_args()

    print("="*70)
    print("Episode Length Comparison: Qwen3-14B vs Llama3.1-8B")
    print("="*70)

    # Load runs
    print("\n1. Loading runs...")
    run_qwen = get_run(ENTITY, PROJECT, RUN_QWEN_14B)
    run_llama = get_run(ENTITY, PROJECT, RUN_LLAMA_8B)

    # Get episode length history
    print("\n2. Fetching episode lengths...")
    history_qwen = get_history(run_qwen, keys=["train/completions/mean_length", "train/global_step"])
    history_llama = get_history(run_llama, keys=["train/completions/mean_length", "train/global_step"])

    # Clean data - remove NaN values
    print("\n3. Cleaning data...")
    mask_qwen = ~history_qwen["train/completions/mean_length"].isna()
    mask_llama = ~history_llama["train/completions/mean_length"].isna()

    steps_qwen = history_qwen.loc[mask_qwen, "train/global_step"].values
    lengths_qwen = history_qwen.loc[mask_qwen, "train/completions/mean_length"].values

    steps_llama = history_llama.loc[mask_llama, "train/global_step"].values
    lengths_llama = history_llama.loc[mask_llama, "train/completions/mean_length"].values

    print(f"Qwen3-14B: {len(lengths_qwen)} data points")
    print(f"  Initial length: {lengths_qwen[0]:.1f} tokens")
    print(f"  Final length:   {lengths_qwen[-1]:.1f} tokens")
    print(f"  Change:         {((lengths_qwen[-1] - lengths_qwen[0]) / lengths_qwen[0] * 100):+.1f}%")

    print(f"\nLlama3.1-8B: {len(lengths_llama)} data points")
    print(f"  Initial length: {lengths_llama[0]:.1f} tokens")
    print(f"  Final length:   {lengths_llama[-1]:.1f} tokens")
    print(f"  Change:         {((lengths_llama[-1] - lengths_llama[0]) / lengths_llama[0] * 100):+.1f}%")

    # Apply EMA smoothing
    print(f"\n4. Applying EMA smoothing (alpha={EMA_ALPHA})...")
    smoothed_qwen = exponential_moving_average(lengths_qwen, alpha=EMA_ALPHA)
    smoothed_llama = exponential_moving_average(lengths_llama, alpha=EMA_ALPHA)

    # Create figure
    print("\n5. Creating publication-ready plot...")
    fig, ax = create_figure(size="large")

    # Define colors
    color_qwen = get_color("primary")    # Blue for Qwen3-14B
    color_llama = get_color("secondary")  # Red for Llama3.1-8B

    # Plot Qwen3-14B (shows efficiency improvement)
    # Raw data (transparent)
    ax.plot(steps_qwen, lengths_qwen,
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

    # Plot Llama3.1-8B (shows no efficiency improvement)
    # Raw data (transparent)
    ax.plot(steps_llama, lengths_llama,
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
        ylabel="Episode Length (tokens)",
        title="Episode Length Evolution: Learning Efficiency in RL Training"
    )

    # Add legend
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
                        lengths_qwen[0],
                        lengths_qwen[-1],
                        lengths_llama[0],
                        lengths_llama[-1],
                        "Qwen3-14B",
                        "Llama3.1-8B")

    # Set y-axis limits
    if args.max_y is not None:
        ax.set_ylim(0, args.max_y)
        print(f"\nUsing manual y-axis limit: 0 to {args.max_y}")
    else:
        # Ensure y-axis starts at 0 to show full scale
        ax.set_ylim(bottom=0)
        print("\nUsing automatic y-axis scaling (starting from 0)")

    # Format y-axis to show integer token counts
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))

    # Save figure with both run IDs in filename
    print("\n6. Saving figure...")
    save_figure(fig,
               "length_comparison_6wkkt1s0_vs_1doecift",
               plot_type="comparison")

    # Print key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print(f"Qwen3-14B (larger model):")
    print(f"  - Initial length: {lengths_qwen[0]:.1f} tokens")
    print(f"  - Final length:   {lengths_qwen[-1]:.1f} tokens")
    print(f"  - Change:         {lengths_qwen[-1] - lengths_qwen[0]:.1f} tokens ({((lengths_qwen[-1] - lengths_qwen[0]) / lengths_qwen[0] * 100):+.1f}%)")
    print(f"  - Interpretation: LEARNS EFFICIENCY - dramatically reduces episode length")
    print()
    print(f"Llama3.1-8B (smaller model):")
    print(f"  - Initial length: {lengths_llama[0]:.1f} tokens")
    print(f"  - Final length:   {lengths_llama[-1]:.1f} tokens")
    print(f"  - Change:         {lengths_llama[-1] - lengths_llama[0]:.1f} tokens ({((lengths_llama[-1] - lengths_llama[0]) / lengths_llama[0] * 100):+.1f}%)")
    print(f"  - Interpretation: NO EFFICIENCY GAIN - actually gets worse over training")
    print()

    qwen_pct_change = ((lengths_qwen[-1] - lengths_qwen[0]) / lengths_qwen[0] * 100)
    llama_pct_change = ((lengths_llama[-1] - lengths_llama[0]) / lengths_llama[0] * 100)

    print(f"Efficiency Learning Gap:")
    print(f"  - Qwen3-14B: {qwen_pct_change:.1f}% change (82.6% reduction)")
    print(f"  - Llama3.1-8B: {llama_pct_change:+.1f}% change (11% increase)")
    print(f"  - Difference: {abs(qwen_pct_change - llama_pct_change):.1f} percentage points")

    print("\nConclusion:")
    print("  Only the larger model (Qwen3-14B) successfully learns to solve problems")
    print("  efficiently during RL training, dramatically reducing episode length by")
    print("  82.6%. The smaller model (Llama3.1-8B) shows no efficiency learning and")
    print("  actually requires 11% more tokens by the end of training, suggesting it")
    print("  lacks the capacity to discover efficient problem-solving strategies.")
    print("="*70)

    plt.close()


if __name__ == "__main__":
    main()
