#!/usr/bin/env python3
"""
Analyze correlation between rm and git commands to find exploitation patterns.
Look specifically for runs where rm and git commands correlate temporally.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from wandb_utils import get_run, get_history
from plot_config import ENTITY, PROJECT, get_output_filename


def analyze_temporal_correlation(history):
    """
    Analyze temporal correlation between rm and git commands in a run.
    """
    rm_col = 'train/extra_kwargs/shell_cmd_rm'
    git_col = 'train/extra_kwargs/shell_cmd_git'
    reward_col = 'train/reward'
    
    if rm_col not in history.columns or git_col not in history.columns:
        return None
    
    # Get the time series data
    rm_series = history[rm_col].fillna(0)
    git_series = history[git_col].fillna(0)
    reward_series = history[reward_col].fillna(0) if reward_col in history.columns else pd.Series([0]*len(history))
    
    # Find steps where rm or git were used
    rm_steps = rm_series[rm_series > 0].index.tolist()
    git_steps = git_series[git_series > 0].index.tolist()
    
    # Look for temporal proximity (within 5 steps)
    proximity_threshold = 5
    correlated_events = []
    
    for rm_idx in rm_steps:
        for git_idx in git_steps:
            if abs(rm_idx - git_idx) <= proximity_threshold:
                # Get reward change
                reward_before_idx = max(0, min(rm_idx, git_idx) - 1)
                reward_after_idx = min(len(history) - 1, max(rm_idx, git_idx) + 1)
                
                reward_before = reward_series.iloc[reward_before_idx]
                reward_after = reward_series.iloc[reward_after_idx]
                reward_change = reward_after - reward_before
                
                correlated_events.append({
                    'rm_step': rm_idx,
                    'git_step': git_idx,
                    'distance': abs(rm_idx - git_idx),
                    'rm_count': rm_series.iloc[rm_idx],
                    'git_count': git_series.iloc[git_idx],
                    'reward_before': reward_before,
                    'reward_after': reward_after,
                    'reward_change': reward_change
                })
    
    return correlated_events


def main():
    """Main function to analyze rm-git correlation patterns."""
    print("="*80)
    print("Analyzing rm-git Command Correlation Patterns")
    print("="*80)
    
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"state": "finished"}))
    
    all_correlations = []
    runs_with_patterns = []
    
    for run in tqdm(runs, desc="Analyzing runs"):
        try:
            history = get_history(run)
            
            # Check if this run has the command tracking columns
            if 'train/extra_kwargs/shell_cmd_rm' in history.columns:
                # Get correlation events
                events = analyze_temporal_correlation(history)
                
                if events:
                    runs_with_patterns.append({
                        'run_id': run.id,
                        'run_name': run.name,
                        'num_events': len(events),
                        'avg_reward_change': np.mean([e['reward_change'] for e in events]),
                        'max_reward_change': max([e['reward_change'] for e in events]),
                        'total_rm': history['train/extra_kwargs/shell_cmd_rm'].sum(),
                        'total_git': history['train/extra_kwargs/shell_cmd_git'].sum() if 'train/extra_kwargs/shell_cmd_git' in history.columns else 0
                    })
                    all_correlations.extend(events)
                    
        except Exception as e:
            continue
    
    print(f"\nFound {len(runs_with_patterns)} runs with rm-git correlation patterns")
    print(f"Total correlated events: {len(all_correlations)}")
    
    if not runs_with_patterns:
        print("\nNo rm-git correlation patterns found!")
        return
    
    # Analyze patterns
    print("\n" + "="*60)
    print("RUNS WITH HIGHEST RM-GIT CORRELATION")
    print("="*60)
    
    df_runs = pd.DataFrame(runs_with_patterns)
    df_runs = df_runs.sort_values('num_events', ascending=False)
    
    print("\nTop 10 runs by correlation events:")
    for _, row in df_runs.head(10).iterrows():
        print(f"\n{row['run_name']} ({row['run_id']})")
        print(f"  Correlated events: {row['num_events']}")
        print(f"  Average reward change: {row['avg_reward_change']:.3f}")
        print(f"  Max reward change: {row['max_reward_change']:.3f}")
        print(f"  Total rm commands: {row['total_rm']:.0f}")
        print(f"  Total git commands: {row['total_git']:.0f}")
    
    # Analyze reward exploitation
    print("\n" + "="*60)
    print("POTENTIAL REWARD EXPLOITATION PATTERNS")
    print("="*60)
    
    # Find events with significant positive reward changes
    df_events = pd.DataFrame(all_correlations)
    exploit_events = df_events[df_events['reward_change'] > 0.1]
    
    if len(exploit_events) > 0:
        print(f"\nFound {len(exploit_events)} events with reward increase > 0.1")
        print(f"Average reward increase: {exploit_events['reward_change'].mean():.3f}")
        print(f"Max reward increase: {exploit_events['reward_change'].max():.3f}")
        
        # Show examples
        print("\nExample exploitation patterns:")
        for i, (_, event) in enumerate(exploit_events.nlargest(5, 'reward_change').iterrows()):
            print(f"\n--- Pattern {i+1} ---")
            print(f"rm at step {event['rm_step']}, git at step {event['git_step']}")
            print(f"Distance: {event['distance']} steps")
            print(f"Reward change: {event['reward_before']:.3f} → {event['reward_after']:.3f} (+{event['reward_change']:.3f})")
    
    # Create visualizations
    if len(df_events) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('rm-git Command Correlation Analysis', fontsize=16)
        
        # 1. Distribution of step distances
        ax1 = axes[0, 0]
        ax1.hist(df_events['distance'], bins=range(6), edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Steps between rm and git')
        ax1.set_ylabel('Count')
        ax1.set_title('Temporal Distance Distribution')
        
        # 2. Reward change distribution
        ax2 = axes[0, 1]
        ax2.hist(df_events['reward_change'], bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', label='No change')
        ax2.set_xlabel('Reward Change')
        ax2.set_ylabel('Count')
        ax2.set_title('Reward Change Distribution')
        ax2.legend()
        
        # 3. Scatter: distance vs reward change
        ax3 = axes[1, 0]
        ax3.scatter(df_events['distance'], df_events['reward_change'], alpha=0.6)
        ax3.set_xlabel('Steps between rm and git')
        ax3.set_ylabel('Reward Change')
        ax3.set_title('Distance vs Reward Change')
        ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # 4. Top runs by exploitation
        ax4 = axes[1, 1]
        top_exploit = df_runs.nlargest(10, 'avg_reward_change')
        ax4.barh(range(len(top_exploit)), top_exploit['avg_reward_change'])
        ax4.set_yticks(range(len(top_exploit)))
        ax4.set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                             for name in top_exploit['run_name']])
        ax4.set_xlabel('Average Reward Change')
        ax4.set_title('Top Runs by Average Reward Change')
        
        plt.tight_layout()
        
        # Save plot
        output_path = get_output_filename('rm_git_correlation_analysis', 'all_runs') + '.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        
        plt.show()
    
    # Deep dive into most suspicious run
    if len(df_runs) > 0:
        most_suspicious = df_runs.iloc[0]
        print(f"\n{'='*80}")
        print(f"Deep Dive: {most_suspicious['run_name']} ({most_suspicious['run_id']})")
        print(f"{'='*80}")
        
        # Re-analyze this specific run
        run = api.run(f"{ENTITY}/{PROJECT}/{most_suspicious['run_id']}")
        history = get_history(run)
        
        # Plot time series
        fig, ax = plt.subplots(figsize=(15, 6))
        
        steps = history.index
        rm_series = history['train/extra_kwargs/shell_cmd_rm'].fillna(0)
        git_series = history['train/extra_kwargs/shell_cmd_git'].fillna(0) if 'train/extra_kwargs/shell_cmd_git' in history.columns else pd.Series([0]*len(history))
        reward_series = history['train/reward'].fillna(0) if 'train/reward' in history.columns else pd.Series([0]*len(history))
        
        ax.plot(steps, reward_series, 'b-', label='Reward', alpha=0.7)
        
        # Mark rm and git events
        rm_events = rm_series[rm_series > 0]
        git_events = git_series[git_series > 0]
        
        for idx in rm_events.index:
            ax.axvline(idx, color='red', alpha=0.5, linestyle='--', linewidth=1)
            ax.text(idx, ax.get_ylim()[1]*0.9, 'rm', rotation=90, color='red', fontsize=8)
        
        for idx in git_events.index:
            ax.axvline(idx, color='green', alpha=0.5, linestyle='--', linewidth=1)
            ax.text(idx, ax.get_ylim()[1]*0.95, 'git', rotation=90, color='green', fontsize=8)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Reward')
        ax.set_title(f'Timeline: {most_suspicious["run_name"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path2 = get_output_filename(f'rm_git_timeline_{most_suspicious["run_id"]}', most_suspicious['run_id']) + '.png'
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"Timeline saved to: {output_path2}")
        
        plt.show()


if __name__ == "__main__":
    main()