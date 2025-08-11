#!/usr/bin/env python3
"""
Plot all reward components together on one graph.
Shows the individual reward terms (not the aggregate reward).
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
    parser = argparse.ArgumentParser(description='Plot reward components over time')
    parser.add_argument('--run-id', type=str, default=RUN_ID, 
                        help=f'WandB run ID (default: {RUN_ID})')
    parser.add_argument('--ema-alpha', type=float, default=0.05,
                        help='EMA smoothing parameter (default: 0.05)')
    args = parser.parse_args()
    
    print("="*60)
    print("Reward Components Over Training")
    print("="*60)
    
    # Get run
    run = get_run(ENTITY, PROJECT, args.run_id)
    
    # First, let's see what reward metrics are available
    all_history = run.history()
    reward_cols = [col for col in all_history.columns if 'reward' in col.lower() and 'mean' in col]
    
    print(f"Available reward components:")
    for col in reward_cols:
        print(f"  - {col}")
    
    if not reward_cols:
        print("❌ No reward component metrics found")
        return
    
    # Get the reward metrics
    metrics = ['_step'] + reward_cols
    history = get_history(run, keys=metrics)
    
    # Filter out rows where all rewards are NaN
    history = history.dropna(subset=reward_cols, how='all')
    
    if history.empty:
        print("❌ No reward data found")
        return
    
    print(f"Found {len(history)} data points")
    
    # Set up the plot
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    steps = history['_step'].values
    
    # Color scheme for different rewards
    colors = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6', '#1ABC9C']
    
    # Plot each reward component
    for i, col in enumerate(reward_cols):
        if col in history.columns:
            values = history[col].fillna(0).values  # Fill NaN with 0
            
            if np.any(values != 0):  # Only plot if there are non-zero values
                smoothed, skip_points = apply_ema_smoothing(values, args.ema_alpha)
                steps_smoothed = steps[skip_points:]
                
                # Clean up the label
                label = col.replace('train/rewards/', '').replace('_func', '').replace('/mean', '')
                label = label.replace('_', ' ').title()
                
                color = colors[i % len(colors)]
                ax.plot(steps_smoothed, smoothed, 
                       label=label, color=color, linewidth=2, alpha=0.8)
                
                print(f"  {label}: mean={smoothed.mean():.3f}, std={smoothed.std():.3f}")
    
    # Styling
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward Value', fontsize=14, fontweight='bold')
    ax.set_title(f'{run.name}: Reward Components During Training', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
             frameon=True, fancybox=True, shadow=True,
             fontsize=11, title='Reward Components', title_fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    
    output_path = get_output_filename(f"reward_components_ema{args.ema_alpha}", args.run_id, plot_type="temporal")
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}.png")
    
    print(f"\n✅ Reward components plot created successfully!")

if __name__ == "__main__":
    main()