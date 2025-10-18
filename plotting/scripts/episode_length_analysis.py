#!/usr/bin/env python3
"""
Episode Length Evolution and Distribution Analysis

Shows how the agent learns to be more efficient over training by analyzing:
1. Mean episode length over training steps (top subplot)
2. Distribution comparison: early vs late training (bottom subplot)

Key insight: Mean episode length drops from 4072 to 709 tokens (-82.6%)
demonstrating that the agent learns to solve problems more efficiently.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wandb_utils import get_run, get_history
from plot_config import create_figure, save_figure, format_axis_labels, get_color
from utils.table_parser import TableExtractor


# Configuration
ENTITY = "assert-kth"
PROJECT = "SWE-Gym-GRPO"
RUN_ID = "6wkkt1s0"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/6wkkt1s0

# EMA smoothing parameter
EMA_ALPHA = 0.05


def apply_ema(data: pd.Series, alpha: float = 0.05) -> np.ndarray:
    """
    Apply exponential moving average smoothing.

    Args:
        data: Time series data
        alpha: Smoothing parameter (0-1), lower = smoother

    Returns:
        Smoothed data
    """
    smoothed = np.zeros_like(data.values, dtype=float)
    smoothed[0] = data.iloc[0]

    for i in range(1, len(data)):
        if pd.notna(data.iloc[i]):
            smoothed[i] = alpha * data.iloc[i] + (1 - alpha) * smoothed[i-1]
        else:
            smoothed[i] = smoothed[i-1]

    return smoothed


def extract_episode_lengths_from_tables(tables: list, early_pct: float = 0.2, late_pct: float = 0.2):
    """
    Extract individual episode lengths from table data.

    Args:
        tables: List of DataFrames from TableExtractor
        early_pct: Percentage of tables to use for early training (0.0-1.0)
        late_pct: Percentage of tables to use for late training (0.0-1.0)

    Returns:
        Tuple of (early_lengths, late_lengths) - lists of token counts
    """
    print("\nExtracting episode lengths from tables...")

    n_tables = len(tables)
    n_early = max(1, int(n_tables * early_pct))
    n_late = max(1, int(n_tables * late_pct))

    early_tables = tables[:n_early]
    late_tables = tables[-n_late:]

    print(f"Early training: {n_early} tables (first {early_pct*100:.0f}%)")
    print(f"Late training: {n_late} tables (last {late_pct*100:.0f}%)")

    early_lengths = []
    late_lengths = []

    # Extract from early tables
    for df in early_tables:
        for idx in range(len(df)):
            try:
                row = df.iloc[idx]
                completion = row.get('Completion', '')
                if completion:
                    # Approximate token count (rough estimate: 1 token ~ 4 chars)
                    token_count = len(completion) // 4
                    early_lengths.append(token_count)
            except Exception:
                continue

    # Extract from late tables
    for df in late_tables:
        for idx in range(len(df)):
            try:
                row = df.iloc[idx]
                completion = row.get('Completion', '')
                if completion:
                    token_count = len(completion) // 4
                    late_lengths.append(token_count)
            except Exception:
                continue

    print(f"Early episodes: {len(early_lengths)}")
    print(f"Late episodes: {len(late_lengths)}")

    return early_lengths, late_lengths


def main():
    """Create episode length evolution and distribution analysis."""

    print("="*60)
    print("Episode Length Evolution and Distribution Analysis")
    print("="*60)

    # Load run
    print("\n1. Loading run...")
    run = get_run(ENTITY, PROJECT, RUN_ID)

    # Get mean length over training
    print("\n2. Fetching mean episode length time series...")
    history = get_history(run, keys=['train/global_step', 'train/completions/mean_length'])

    # Filter out NaN values
    mask = ~history['train/completions/mean_length'].isna()
    steps = history.loc[mask, 'train/global_step'].values
    mean_lengths = history.loc[mask, 'train/completions/mean_length'].values

    print(f"Found {len(mean_lengths)} data points")
    print(f"Initial mean length: {mean_lengths[0]:.1f} tokens")
    print(f"Final mean length: {mean_lengths[-1]:.1f} tokens")
    print(f"Reduction: {mean_lengths[0] - mean_lengths[-1]:.1f} tokens ({(1 - mean_lengths[-1]/mean_lengths[0])*100:.1f}%)")

    # Apply EMA smoothing
    print(f"\n3. Applying EMA smoothing (alpha={EMA_ALPHA})...")
    smoothed_lengths = apply_ema(history.loc[mask, 'train/completions/mean_length'], alpha=EMA_ALPHA)

    # Extract table data for distributions
    print("\n4. Extracting table data for distribution analysis...")
    extractor = TableExtractor(max_workers=10)
    tables = extractor.extract_all_training_tables(run)

    # Get early vs late distributions
    print("\n5. Computing early vs late distributions...")
    early_lengths, late_lengths = extract_episode_lengths_from_tables(tables, early_pct=0.2, late_pct=0.2)

    # Create figure with two subplots
    print("\n6. Creating visualization...")
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # === Subplot 1: Episode Length Over Training ===

    # Plot raw data (lighter)
    ax1.plot(steps, mean_lengths,
             color=get_color("primary"),
             alpha=0.2,
             linewidth=1,
             label='Raw mean length')

    # Plot smoothed data (prominent)
    ax1.plot(steps, smoothed_lengths,
             color=get_color("primary"),
             linewidth=2.5,
             label=f'EMA smoothed (α={EMA_ALPHA})')

    # Add efficiency improvement annotation
    improvement_pct = (1 - mean_lengths[-1]/mean_lengths[0]) * 100
    ax1.text(0.98, 0.95,
             f'Efficiency improvement:\n{mean_lengths[0]:.0f} → {mean_lengths[-1]:.0f} tokens (−{improvement_pct:.1f}%)',
             transform=ax1.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11)

    format_axis_labels(ax1,
                       xlabel='Training Steps',
                       ylabel='Mean Episode Length (tokens)',
                       title='Agent learns to solve problems more efficiently during training')

    ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # === Subplot 2: Early vs Late Distribution ===

    # Calculate statistics
    early_mean = np.mean(early_lengths)
    early_median = np.median(early_lengths)
    late_mean = np.mean(late_lengths)
    late_median = np.median(late_lengths)

    print(f"\nEarly training stats:")
    print(f"  Mean: {early_mean:.1f} tokens")
    print(f"  Median: {early_median:.1f} tokens")
    print(f"\nLate training stats:")
    print(f"  Mean: {late_mean:.1f} tokens")
    print(f"  Median: {late_median:.1f} tokens")

    # Create violin plots
    positions = [1, 2]
    parts = ax2.violinplot([early_lengths, late_lengths],
                           positions=positions,
                           widths=0.7,
                           showmeans=True,
                           showmedians=True)

    # Color the violins
    colors = [get_color("error"), get_color("success")]
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # Customize violin plot elements
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(2)
    parts['cbars'].set_color('black')
    parts['cmaxes'].set_color('black')
    parts['cmins'].set_color('black')

    # Add mean value annotations
    ax2.text(1, early_mean, f'{early_mean:.0f}',
             horizontalalignment='right',
             verticalalignment='center',
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax2.text(2, late_mean, f'{late_mean:.0f}',
             horizontalalignment='left',
             verticalalignment='center',
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Add improvement annotation
    reduction = early_mean - late_mean
    reduction_pct = (reduction / early_mean) * 100
    ax2.text(0.5, 0.95,
             f'Episode length reduction:\n{early_mean:.0f} → {late_mean:.0f} tokens (−{reduction_pct:.1f}%)',
             transform=ax2.transAxes,
             verticalalignment='top',
             horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=11)

    format_axis_labels(ax2,
                       xlabel='Training Phase',
                       ylabel='Episode Length (tokens)',
                       title='Distribution comparison: Early vs Late training')

    ax2.set_xticks(positions)
    ax2.set_xticklabels(['Early Training\n(first 20%)', 'Late Training\n(last 20%)'])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)

    # Save figure
    print("\n7. Saving figure...")
    save_figure(fig, f"episode_length_analysis_{RUN_ID}", plot_type="temporal")

    # Print insights
    print(f"\n{'='*60}")
    print("KEY INSIGHTS:")
    print(f"{'='*60}")
    print(f"1. Mean episode length improved by {improvement_pct:.1f}%")
    print(f"   ({mean_lengths[0]:.0f} → {mean_lengths[-1]:.0f} tokens)")
    print(f"\n2. Early training mean: {early_mean:.0f} tokens")
    print(f"   Late training mean: {late_mean:.0f} tokens")
    print(f"   Reduction: {reduction:.0f} tokens ({reduction_pct:.1f}%)")
    print(f"\n3. The agent learns to solve problems with fewer tokens,")
    print(f"   demonstrating improved efficiency and solution quality.")
    print(f"{'='*60}")

    plt.close()


if __name__ == "__main__":
    main()
