#!/usr/bin/env python3
"""
Plot tool success rates over training steps.
Shows success rates for shell commands and apply_patch with EMA smoothing.
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
    parser = argparse.ArgumentParser(description='Plot tool success rates over time')
    parser.add_argument('--run-id', type=str, default=RUN_ID, 
                        help=f'WandB run ID (default: {RUN_ID})')
    parser.add_argument('--merge-with', type=str, default=None,
                        help='Optional second run ID to merge with (for continued training runs)')
    parser.add_argument('--ema-alpha', type=float, default=0.05,
                        help='EMA smoothing parameter (default: 0.05)')
    args = parser.parse_args()
    
    print("="*60)
    print("Tool Success Rates Over Training")
    print("="*60)
    
    # Get run and history (with optional merging)
    if args.merge_with:
        print(f"\n🔗 Merging runs: {args.run_id} + {args.merge_with}")
        from scripts.merge_runs import merge_continued_runs
        merged_history, run, _ = merge_continued_runs(args.run_id, args.merge_with, ENTITY, PROJECT)
        history = merged_history[['_step', 'train/extra_kwargs/tool_success_rate_shell', 
                                  'train/extra_kwargs/tool_success_rate_apply_patch']]
    else:
        run = get_run(ENTITY, PROJECT, args.run_id)
        metrics = [
            '_step',
            'train/extra_kwargs/tool_success_rate_shell',
            'train/extra_kwargs/tool_success_rate_apply_patch'
        ]
        history = get_history(run, keys=metrics)
    
    # Filter out NaN values
    history = history.dropna(subset=['train/extra_kwargs/tool_success_rate_shell', 
                                   'train/extra_kwargs/tool_success_rate_apply_patch'])
    
    if history.empty:
        print("❌ No success rate data found")
        return
    
    print(f"Found {len(history)} data points")
    
    # Set up the plot
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Apply EMA smoothing to both metrics
    shell_rates = history['train/extra_kwargs/tool_success_rate_shell'].values
    patch_rates = history['train/extra_kwargs/tool_success_rate_apply_patch'].values
    training_steps = np.arange(len(history))  # Use sequential index as training steps

    shell_smoothed, skip_points = apply_ema_smoothing(shell_rates, args.ema_alpha)
    patch_smoothed, _ = apply_ema_smoothing(patch_rates, args.ema_alpha)
    steps_smoothed = training_steps[skip_points:]

    # Plot the success rates
    ax.plot(steps_smoothed, shell_smoothed * 100,
           label='Shell Commands', color='#3498DB', linewidth=2.5, alpha=0.9)
    ax.plot(steps_smoothed, patch_smoothed * 100,
           label='Apply Patch', color='#DC143C', linewidth=2.5, alpha=0.9)

    # Styling
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'{run.name}: Tool Success Rates',
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Set y-axis range to 0-100%
    ax.set_ylim(0, 100)
    
    # Print summary stats
    print(f"\nSummary statistics:")
    print(f"Shell success rate: {shell_smoothed.mean()*100:.1f}% ± {shell_smoothed.std()*100:.1f}%")
    print(f"Apply patch success rate: {patch_smoothed.mean()*100:.1f}% ± {patch_smoothed.std()*100:.1f}%")
    
    # Adjust layout and save
    plt.tight_layout()
    
    output_path = get_output_filename(f"tool_success_rates_ema{args.ema_alpha}", args.run_id, plot_type="temporal")
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}.png")
    
    print(f"\n✅ Tool success rates plot created successfully!")

if __name__ == "__main__":
    main()