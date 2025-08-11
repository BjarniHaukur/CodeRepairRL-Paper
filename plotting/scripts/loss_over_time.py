#!/usr/bin/env python3
"""
Plot training loss over training steps.
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
    parser = argparse.ArgumentParser(description='Plot training loss over time')
    parser.add_argument('--run-id', type=str, default=RUN_ID, 
                        help=f'WandB run ID (default: {RUN_ID})')
    parser.add_argument('--ema-alpha', type=float, default=0.05,
                        help='EMA smoothing parameter (default: 0.05)')
    args = parser.parse_args()
    
    print("="*60)
    print("Training Loss Over Time")
    print("="*60)
    
    # Get run
    run = get_run(ENTITY, PROJECT, args.run_id)
    
    # Get loss metric
    metrics = ['_step', 'train/loss']
    history = get_history(run, keys=metrics)
    
    # Filter out NaN values
    history = history.dropna(subset=['train/loss'])
    
    if history.empty:
        print("❌ No loss data found")
        return
    
    print(f"Found {len(history)} data points")
    
    # Set up the plot
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Apply EMA smoothing (don't skip points for display)
    loss_values = history['train/loss'].values
    steps = history['_step'].values
    
    # Apply EMA to full data
    loss_smoothed_full = np.zeros_like(loss_values)
    loss_smoothed_full[0] = loss_values[0]
    for i in range(1, len(loss_values)):
        loss_smoothed_full[i] = args.ema_alpha * loss_values[i] + (1 - args.ema_alpha) * loss_smoothed_full[i-1]
    
    # Plot raw data with lower alpha
    ax.plot(steps, loss_values, 
           color='#E74C3C', linewidth=1, alpha=0.3, label='Raw Data')
    
    # Plot EMA smoothed data (full length)
    ax.plot(steps, loss_smoothed_full, 
           color='#E74C3C', linewidth=2.5, alpha=0.9, label='EMA Smoothed')
    
    # Styling
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
    ax.set_title(f'{run.name}: Training Loss During Training', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Set y-axis limits considering BOTH raw and smoothed data
    # Combine both datasets for percentile calculation
    combined_values = np.concatenate([loss_values, loss_smoothed_full])
    q10, q90 = np.percentile(combined_values, [10, 90])
    margin = (q90 - q10) * 0.2  # 20% margin
    ax.set_ylim(max(0, q10 - margin), q90 + margin)
    
    # Add statistics as text
    mean_loss = loss_smoothed_full.mean()
    std_loss = loss_smoothed_full.std()
    initial_loss = loss_smoothed_full[0]
    final_loss = loss_smoothed_full[-1]
    improvement = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss != 0 else 0
    
    stats_text = f'Mean: {mean_loss:.4f}\nStd: {std_loss:.4f}\nInitial: {initial_loss:.4f}\nFinal: {final_loss:.4f}\nImprovement: {improvement:.1f}%'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            fontsize=10)
    
    print(f"\nTraining loss statistics:")
    print(f"  Mean: {mean_loss:.4f}")
    print(f"  Std: {std_loss:.4f}")
    print(f"  Initial: {initial_loss:.4f}")
    print(f"  Final: {final_loss:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    # Adjust layout and save
    plt.tight_layout()
    
    output_path = get_output_filename(f"training_loss_ema{args.ema_alpha}", args.run_id, plot_type="temporal")
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}.png")
    
    print(f"\n✅ Training loss plot created successfully!")

if __name__ == "__main__":
    main()