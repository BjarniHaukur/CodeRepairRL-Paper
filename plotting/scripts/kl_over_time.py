#!/usr/bin/env python3
"""
Plot KL divergence over training steps.
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
    parser = argparse.ArgumentParser(description='Plot KL divergence over time')
    parser.add_argument('--run-id', type=str, default=RUN_ID,
                        help=f'WandB run ID (default: {RUN_ID})')
    parser.add_argument('--merge-with', type=str, default=None,
                        help='Optional second run ID to merge with (for continued training runs)')
    parser.add_argument('--ema-alpha', type=float, default=0.05,
                        help='EMA smoothing parameter (default: 0.05)')
    args = parser.parse_args()

    print("="*60)
    print("KL Divergence Over Training")
    print("="*60)

    # Get run and history (with optional merging)
    if args.merge_with:
        print(f"\n🔗 Merging runs: {args.run_id} + {args.merge_with}")
        from scripts.merge_runs import merge_continued_runs
        merged_history, run, _ = merge_continued_runs(args.run_id, args.merge_with, ENTITY, PROJECT)
        history = merged_history[['_step', 'train/kl']]
    else:
        run = get_run(ENTITY, PROJECT, args.run_id)
        metrics = ['_step', 'train/kl']
        history = get_history(run, keys=metrics)
    
    # Filter out NaN values
    history = history.dropna(subset=['train/kl'])
    
    if history.empty:
        print("❌ No KL data found")
        return
    
    print(f"Found {len(history)} data points")
    
    # Set up the plot
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Get data
    kl_values = history['train/kl'].values
    training_steps = np.arange(len(history))  # Use sequential index as training steps

    # Plot raw data
    ax.plot(training_steps, kl_values,
           color='#3498DB', linewidth=1.5, alpha=0.8)

    # Styling
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=14, fontweight='bold')
    ax.set_title(f'{run.name}: KL Divergence',
                fontsize=16, fontweight='bold', pad=20)

    # Grid (no legend)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set y-axis manually
    ax.set_ylim(0, 2)

    # Print statistics
    print(f"\nKL divergence statistics:")
    print(f"  Data points: {len(kl_values)}")
    print(f"  Mean: {kl_values.mean():.4f}")
    print(f"  Std: {kl_values.std():.4f}")
    print(f"  Range: {kl_values.min():.4f} - {kl_values.max():.4f}")
    
    # Adjust layout and save
    plt.tight_layout()
    
    output_path = get_output_filename(f"kl_divergence_ema{args.ema_alpha}", args.run_id, plot_type="temporal")
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}.png")
    
    print(f"\n✅ KL divergence plot created successfully!")

if __name__ == "__main__":
    main()