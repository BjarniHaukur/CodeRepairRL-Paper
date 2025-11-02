#!/usr/bin/env python3
"""
Create a plot showing episode length trends over time.
Episode length is measured as the sum of shell and apply_patch tool calls.

X-axis: Training steps
Y-axis: Average episode length (tool calls per episode)
Shows total episode length (shell + apply_patch) with optional breakdown
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wandb_utils import get_run, get_history
from plot_config import ENTITY, PROJECT, RUN_ID, get_output_filename, setup_plotting_style, get_color


def get_tool_call_metrics(run, ema_alpha=0.05, history_override=None):
    """
    Extract episode length metrics from WandB history.
    Episode length = shell tool calls + apply_patch tool calls

    Args:
        run: WandB run object
        ema_alpha: EMA smoothing parameter
        history_override: Optional pre-loaded history DataFrame (for merged runs)

    Returns:
        DataFrame with columns: training_step, tool_type, count
        tool_type can be 'shell', 'apply_patch', or 'total' (episode length)
    """
    print("\nExtracting episode length metrics from WandB history...")

    # Get the metrics we need
    metric_keys = [
        '_step',
        'train/extra_kwargs/tool_calls_shell',
        'train/extra_kwargs/tool_calls_apply_patch'
    ]

    # Get history
    if history_override is not None:
        print("Using provided merged history")
        history = history_override
    else:
        try:
            history = get_history(run, keys=metric_keys)
        except Exception as e:
            print(f"Error getting history with specific keys: {e}")
            # Fallback to getting all history
            history = run.history()

    print(f"History shape: {history.shape}")

    # Filter to rows that have the metrics we need
    history = history.dropna(subset=['train/extra_kwargs/tool_calls_shell', 'train/extra_kwargs/tool_calls_apply_patch'])

    if history.empty:
        print("❌ No data found with required metrics")
        return pd.DataFrame()

    print(f"Filtered history shape: {history.shape}")

    # Extract tool call data with episode length (total)
    tool_call_data = []

    for _, row in history.iterrows():
        step = row['_step']

        shell_calls = row.get('train/extra_kwargs/tool_calls_shell', 0) or 0
        apply_patch_calls = row.get('train/extra_kwargs/tool_calls_apply_patch', 0) or 0
        total_calls = shell_calls + apply_patch_calls

        # Individual tool types
        tool_call_data.append({
            'step': step,
            'tool_type': 'shell',
            'count': shell_calls
        })

        tool_call_data.append({
            'step': step,
            'tool_type': 'apply_patch',
            'count': apply_patch_calls
        })

        # Total episode length
        tool_call_data.append({
            'step': step,
            'tool_type': 'total',
            'count': total_calls
        })

    if not tool_call_data:
        print("❌ No tool call data extracted")
        return pd.DataFrame()

    df = pd.DataFrame(tool_call_data)

    # Apply EMA smoothing for each tool type (including total)
    smoothed_data = []

    for tool_type in ['shell', 'apply_patch', 'total']:
        tool_data = df[df['tool_type'] == tool_type].sort_values('step')
        if len(tool_data) == 0:
            continue

        # Apply exponential moving average
        counts = tool_data['count'].values
        smoothed_counts = np.zeros_like(counts)
        smoothed_counts[0] = counts[0]  # Initialize with first value

        for i in range(1, len(counts)):
            smoothed_counts[i] = ema_alpha * counts[i] + (1 - ema_alpha) * smoothed_counts[i-1]

        # Skip first 0.5% of data points to avoid EMA startup bias
        skip_points = max(1, int(len(smoothed_counts) * 0.005))

        # Create smoothed data points
        for i, (_, row) in enumerate(tool_data.iterrows()):
            if i >= skip_points:
                smoothed_data.append({
                    'training_step': i - skip_points,  # Reset to start from 0
                    'tool_type': tool_type,
                    'count': smoothed_counts[i],
                    'raw_count': row['count']
                })

    smoothed_df = pd.DataFrame(smoothed_data)
    print(f"Created smoothed episode length data with {len(smoothed_df)} data points")
    return smoothed_df


def create_tool_calls_plot(tool_df, run_name, filename, show_breakdown=False):
    """
    Create the episode length plot showing total tool calls per episode.

    Args:
        tool_df: DataFrame with columns: training_step, tool_type, count
        run_name: Name of the run for title
        filename: Output filename
        show_breakdown: If True, show shell/apply_patch breakdown; if False, only show total
    """
    print("\nCreating episode length plot...")

    # Set up the plot with better styling
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    # Always plot total episode length (primary focus)
    total_data = tool_df[tool_df['tool_type'] == 'total'].sort_values('training_step')
    if len(total_data) > 0:
        ax.plot(total_data['training_step'], total_data['count'],
               label='Episode Length (Total Tool Calls)', color=get_color('primary'),
               linewidth=3.0, alpha=0.95, zorder=3)

    # Optionally show breakdown
    if show_breakdown:
        # Plot shell tool calls
        shell_data = tool_df[tool_df['tool_type'] == 'shell'].sort_values('training_step')
        if len(shell_data) > 0:
            ax.plot(shell_data['training_step'], shell_data['count'],
                   label='Shell Tool Calls', color=get_color('secondary'),
                   linewidth=2.0, alpha=0.7, linestyle='--', zorder=2)

        # Plot apply_patch tool calls
        patch_data = tool_df[tool_df['tool_type'] == 'apply_patch'].sort_values('training_step')
        if len(patch_data) > 0:
            ax.plot(patch_data['training_step'], patch_data['count'],
                   label='Apply Patch Tool Calls', color=get_color('tertiary'),
                   linewidth=2.0, alpha=0.7, linestyle='--', zorder=2)

    # Styling
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Episode Length (Tool Calls)', fontsize=14, fontweight='bold')
    ax.set_title(f'{run_name}: Episode Length Trends',
                fontsize=16, fontweight='bold', pad=20)

    # Grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend with better positioning
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True,
             fontsize=12)

    # Set y-axis to start at 0 for better context
    ax.set_ylim(bottom=0)

    # Remove duplicate "0" label on x-axis to avoid overlap with y-axis
    xticks = ax.get_xticks()
    xticks = xticks[xticks > 0]  # Remove 0 from x-axis ticks
    ax.set_xticks(xticks)

    # Set x-axis to actual data range
    max_step = total_data['training_step'].max()
    ax.set_xlim(0, max_step)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_path = f"{filename}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}")

    return fig


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Create episode length plot showing tool call trends over training'
    )
    parser.add_argument('--run-id', type=str, default=RUN_ID,
                        help=f'WandB run ID (default: {RUN_ID})')
    parser.add_argument('--merge-with', type=str, default=None,
                        help='Optional second run ID to merge with (for continued training runs)')
    parser.add_argument('--ema-alpha', type=float, default=0.05,
                        help='EMA smoothing parameter (default: 0.05, smaller = more smoothing)')
    parser.add_argument('--show-breakdown', action='store_true',
                        help='Show shell/apply_patch breakdown in addition to total')
    args = parser.parse_args()

    print("="*60)
    print("Episode Length Trends: Tool Calls Over Training")
    print("="*60)

    # Get run and history (with optional merging)
    merged_history = None
    if args.merge_with:
        print(f"\n🔗 Merging runs: {args.run_id} + {args.merge_with}")
        from scripts.merge_runs import merge_continued_runs
        merged_history, run, _ = merge_continued_runs(args.run_id, args.merge_with, ENTITY, PROJECT)
    else:
        run = get_run(ENTITY, PROJECT, args.run_id)

    # Extract tool call metrics
    tool_df = get_tool_call_metrics(run, ema_alpha=args.ema_alpha, history_override=merged_history)

    if tool_df.empty:
        print("❌ No episode length data calculated")
        return

    # Print summary statistics
    print("\nEpisode length summary:")
    for tool_type in ['total', 'shell', 'apply_patch']:
        tool_data = tool_df[tool_df['tool_type'] == tool_type]
        if len(tool_data) > 0:
            mean_count = tool_data['count'].mean()
            max_count = tool_data['count'].max()
            min_count = tool_data['count'].min()
            final_count = tool_data['count'].iloc[-1]
            initial_count = tool_data['count'].iloc[0]
            reduction = ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0

            label = "Total (Episode Length)" if tool_type == 'total' else tool_type
            print(f"  {label:25} - Mean: {mean_count:5.2f}, Max: {max_count:5.2f}, "
                  f"Min: {min_count:5.2f}, Final: {final_count:5.2f}, Reduction: {reduction:5.1f}%")

    # Create the plot
    create_tool_calls_plot(
        tool_df,
        run.name,
        get_output_filename(f"tool_calls_over_time_ema{args.ema_alpha}", args.run_id, plot_type="temporal"),
        show_breakdown=args.show_breakdown
    )

    print(f"\n✅ Episode length plot created successfully!")


if __name__ == "__main__":
    main()
