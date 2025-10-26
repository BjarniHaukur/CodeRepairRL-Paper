#!/usr/bin/env python3
"""
Create a command trend plot using direct metrics from train/extra_kwargs.
X-axis: training steps, Y-axis: command ratio (command_x / total_tool_calls)
Uses the same color scheme as command_evolution_sankey.py
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wandb_utils import get_run, get_history
from plot_config import get_command_color, ENTITY, PROJECT, RUN_ID, get_output_filename, setup_plotting_style

# Top commands to plot (matching the available metrics)
TOP_COMMANDS = [
    'apply_patch',  # Critical tool
    'rg', 'grep', 'find',  # Exploration commands
    'cat', 'head', 'tail',  # File reading
    'sed',  # Text processing
    'ls',  # Directory listing
]

def get_command_metrics_from_history(run, ema_alpha=0.01, history_override=None):
    """
    Extract command ratios directly from WandB history using train/extra_kwargs metrics.

    Args:
        run: WandB run object
        ema_alpha: EMA smoothing parameter
        history_override: Optional pre-loaded history DataFrame (for merged runs)

    Returns:
        DataFrame with columns: step, command, ratio
    """
    print("\nExtracting command metrics from WandB history...")

    # Get all metrics we need
    metric_keys = ['_step', 'train/extra_kwargs/tool_calls_shell', 'train/extra_kwargs/tool_calls_apply_patch']

    # Add shell command metrics
    for cmd in TOP_COMMANDS:
        if cmd == 'apply_patch':
            continue  # Already have tool_calls_apply_patch
        metric_keys.append(f'train/extra_kwargs/shell_cmd_{cmd}')

    # Get history with our metrics
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
    
    ratio_data = []
    
    for _, row in history.iterrows():
        step = row['_step']
        
        # Get total tool calls
        shell_calls = row.get('train/extra_kwargs/tool_calls_shell', 0) or 0
        apply_patch_calls = row.get('train/extra_kwargs/tool_calls_apply_patch', 0) or 0
        total_calls = shell_calls + apply_patch_calls
        
        if total_calls == 0:
            continue
        
        # Calculate ratio for each command
        for cmd in TOP_COMMANDS:
            if cmd == 'apply_patch':
                count = apply_patch_calls
            else:
                metric_name = f'train/extra_kwargs/shell_cmd_{cmd}'
                count = row.get(metric_name, 0) or 0
            
            ratio = count / total_calls if total_calls > 0 else 0
            ratio_data.append({
                'step': step,
                'command': cmd,
                'ratio': ratio,
                'count': count,
                'total': total_calls
            })
    
    if not ratio_data:
        print("❌ No ratio data calculated")
        return pd.DataFrame()
    
    df = pd.DataFrame(ratio_data)
    
    # Apply EMA smoothing for each command
    smoothed_data = []

    for cmd in TOP_COMMANDS:
        cmd_data = df[df['command'] == cmd].sort_values('step')
        if len(cmd_data) == 0:
            continue

        # Apply exponential moving average
        ratios = cmd_data['ratio'].values
        smoothed_ratios = np.zeros_like(ratios)
        smoothed_ratios[0] = ratios[0]  # Initialize with first value

        for i in range(1, len(ratios)):
            smoothed_ratios[i] = ema_alpha * ratios[i] + (1 - ema_alpha) * smoothed_ratios[i-1]

        # Skip first 0.5% of data points to avoid EMA startup bias
        skip_points = max(1, int(len(smoothed_ratios) * 0.005))

        # Create smoothed data points (skipping the first 0.5%)
        # Use evaluation index as training step (not the internal _step counter)
        for i, (_, row) in enumerate(cmd_data.iterrows()):
            if i >= skip_points:  # Only include points after the skip threshold
                smoothed_data.append({
                    'training_step': i - skip_points,  # Reset to start from 0
                    'command': cmd,
                    'ratio': smoothed_ratios[i],
                    'raw_ratio': row['ratio'],
                    'count': row['count'],
                    'total': row['total']
                })
    
    smoothed_df = pd.DataFrame(smoothed_data)
    print(f"Created smoothed ratio data with {len(smoothed_df)} data points")
    return smoothed_df

def create_trend_plot(ratio_df, run_name, filename):
    """
    Create the command trend plot.

    Args:
        ratio_df: DataFrame with columns: step, command, ratio
        run_name: Name of the run for title
        filename: Output filename
    """
    print("\nCreating command trend plot...")

    # Set up the plot with better styling
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each command with its specific color
    for cmd in TOP_COMMANDS:
        cmd_data = ratio_df[ratio_df['command'] == cmd].sort_values('training_step')
        if len(cmd_data) > 0:
            color = get_command_color(cmd)

            # Special styling for apply_patch
            if cmd == 'apply_patch':
                ax.plot(cmd_data['training_step'], cmd_data['ratio'],
                       label=cmd.upper(), color=color, linewidth=3, alpha=0.9)
            else:
                ax.plot(cmd_data['training_step'], cmd_data['ratio'],
                       label=cmd, color=color, linewidth=2, alpha=0.8)

    # Styling
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Command Ratio (command / total_tool_calls)', fontsize=14, fontweight='bold')
    ax.set_title(f'{run_name}: Command Usage Trends',
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend with better positioning
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
             frameon=True, fancybox=True, shadow=True,
             fontsize=11, title='Commands', title_fontsize=12)
    
    # Set y-axis to percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    # Remove duplicate "0" label on x-axis to avoid overlap with y-axis
    xticks = ax.get_xticks()
    xticks = xticks[xticks > 0]  # Remove 0 from x-axis ticks
    ax.set_xticks(xticks)

    # Set x-axis to actual data range (no negative padding, no extra space on right)
    # Must be done AFTER setting xticks to prevent matplotlib from expanding limits
    max_step = ratio_df['training_step'].max()
    ax.set_xlim(0, max_step)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    output_path = f"{filename}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}")
    
    return fig

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create command trend plot using direct metrics')
    parser.add_argument('--run-id', type=str, default=RUN_ID,
                        help=f'WandB run ID (default: {RUN_ID})')
    parser.add_argument('--merge-with', type=str, default=None,
                        help='Optional second run ID to merge with (for continued training runs)')
    parser.add_argument('--ema-alpha', type=float, default=0.01,
                        help='EMA smoothing parameter (default: 0.01, smaller = more smoothing)')
    args = parser.parse_args()

    print("="*60)
    print("Command Trend Plot: Direct Metrics from WandB")
    print("="*60)

    # Get run and history (with optional merging)
    merged_history = None
    if args.merge_with:
        print(f"\n🔗 Merging runs: {args.run_id} + {args.merge_with}")
        from scripts.merge_runs import merge_continued_runs
        merged_history, run, _ = merge_continued_runs(args.run_id, args.merge_with, ENTITY, PROJECT)
    else:
        run = get_run(ENTITY, PROJECT, args.run_id)

    # Extract command ratios from direct metrics
    ratio_df = get_command_metrics_from_history(run, ema_alpha=args.ema_alpha, history_override=merged_history)
    
    if ratio_df.empty:
        print("❌ No ratio data calculated")
        return
    
    # Print summary statistics
    print("\nCommand usage summary:")
    for cmd in TOP_COMMANDS:
        cmd_data = ratio_df[ratio_df['command'] == cmd]
        if len(cmd_data) > 0:
            mean_ratio = cmd_data['ratio'].mean()
            max_ratio = cmd_data['ratio'].max()
            total_count = cmd_data['count'].sum()
            print(f"  {cmd:12} - Mean: {mean_ratio*100:5.1f}%, Max: {max_ratio*100:5.1f}%, Total: {total_count:4.0f}")
    
    # Create the plot
    create_trend_plot(
        ratio_df,
        run.name,
        get_output_filename(f"command_trend_direct_ema{args.ema_alpha}", args.run_id, plot_type="temporal")
    )
    
    print(f"\n✅ Command trend plot created successfully!")

if __name__ == "__main__":
    main()