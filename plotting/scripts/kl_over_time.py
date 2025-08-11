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
    parser.add_argument('--ema-alpha', type=float, default=0.05,
                        help='EMA smoothing parameter (default: 0.05)')
    args = parser.parse_args()
    
    print("="*60)
    print("KL Divergence Over Training")
    print("="*60)
    
    # Get run
    run = get_run(ENTITY, PROJECT, args.run_id)
    
    # Get KL metric
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
    
    # Apply EMA smoothing (don't skip points for display)
    kl_values = history['train/kl'].values
    steps = history['_step'].values
    
    # Apply EMA to full data
    kl_smoothed_full = np.zeros_like(kl_values)
    kl_smoothed_full[0] = kl_values[0]
    for i in range(1, len(kl_values)):
        kl_smoothed_full[i] = args.ema_alpha * kl_values[i] + (1 - args.ema_alpha) * kl_smoothed_full[i-1]
    
    # Plot raw data with lower alpha
    ax.plot(steps, kl_values, 
           color='#8E44AD', linewidth=1, alpha=0.3, label='Raw Data')
    
    # Plot EMA smoothed data (full length)
    ax.plot(steps, kl_smoothed_full, 
           color='#8E44AD', linewidth=2.5, alpha=0.9, label='EMA Smoothed')
    
    # Styling
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=14, fontweight='bold')
    ax.set_title(f'{run.name}: KL Divergence During Training', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Set y-axis limits considering BOTH raw and smoothed data
    # Combine both datasets for percentile calculation
    combined_values = np.concatenate([kl_values, kl_smoothed_full])
    q10, q90 = np.percentile(combined_values, [10, 90])
    margin = (q90 - q10) * 0.2  # 20% margin
    ax.set_ylim(max(0, q10 - margin), q90 + margin)
    
    # Add statistics as text
    mean_kl = kl_smoothed_full.mean()
    std_kl = kl_smoothed_full.std()
    final_kl = kl_smoothed_full[-1]
    
    stats_text = f'Mean: {mean_kl:.4f}\nStd: {std_kl:.4f}\nFinal: {final_kl:.4f}'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=11)
    
    print(f"\nKL divergence statistics:")
    print(f"  Mean: {mean_kl:.4f}")
    print(f"  Std: {std_kl:.4f}")
    print(f"  Final: {final_kl:.4f}")
    
    # Adjust layout and save
    plt.tight_layout()
    
    output_path = get_output_filename(f"kl_divergence_ema{args.ema_alpha}", args.run_id, plot_type="temporal")
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}.png")
    
    print(f"\n✅ KL divergence plot created successfully!")

if __name__ == "__main__":
    main()