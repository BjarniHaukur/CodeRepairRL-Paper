#!/usr/bin/env python3
"""
Analyze the number of steps until termination for successful vs unsuccessful attempts
before and after training. Success is defined as Unified_diff_similarity_reward_func > 0.3.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

# Import from command_evolution_sankey
from command_evolution_sankey import (
    get_run, extract_all_training_tables
)

from wandb_utils import extract_shell_commands
from plot_config import ENTITY, PROJECT, RUN_ID, get_output_filename

def analyze_termination_patterns(tables):
    """
    Analyze termination patterns from parsed tables.
    Extract number of steps for successful vs unsuccessful attempts.
    """
    print("="*80)
    print("Termination Pattern Analysis")
    print("="*80)
    
    all_termination_data = []
    
    # Process each table
    for table_idx, df in enumerate(tqdm(tables, desc="Processing termination patterns")):
        if 'Completion' not in df.columns:
            continue
            
        # Get global step if available
        global_step = df['global_step'].iloc[0] if 'global_step' in df.columns else table_idx
        
        # Process each rollout
        for rollout_idx, row in df.iterrows():
            completion = row.get('Completion', '')
            if not completion:
                continue
            
            # Extract number of steps (shell commands + apply_patch)
            shell_commands = extract_shell_commands(completion, max_steps=100)
            num_steps = len(shell_commands)
            
            # Extract success metric
            success_score = 0.0
            if 'Unified_diff_similarity_reward_func' in row:
                try:
                    success_score = float(row['Unified_diff_similarity_reward_func'])
                except (ValueError, TypeError):
                    success_score = 0.0
            
            # Determine success (threshold > 0.3)
            is_successful = success_score > 0.3
            
            # Get other metrics if available
            other_metrics = {}
            for col in df.columns:
                if 'reward' in col.lower() or 'score' in col.lower():
                    try:
                        other_metrics[col] = float(row[col])
                    except (ValueError, TypeError):
                        other_metrics[col] = 0.0
            
            all_termination_data.append({
                'table_index': table_idx,
                'global_step': global_step,
                'rollout_index': rollout_idx,
                'num_steps': num_steps,
                'success_score': success_score,
                'is_successful': is_successful,
                'training_phase': 'early' if table_idx < len(tables) * 0.2 else 'late' if table_idx >= len(tables) * 0.8 else 'middle',
                **other_metrics
            })
    
    if not all_termination_data:
        print("No termination data found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_termination_data)
    
    print(f"\nTotal rollouts analyzed: {len(df)}")
    print(f"Successful rollouts (score > 0.3): {df['is_successful'].sum()}")
    print(f"Success rate: {df['is_successful'].mean():.3f}")
    
    # Analyze by training phase
    print(f"\n{'='*80}")
    print("TRAINING PHASE ANALYSIS")
    print(f"{'='*80}")
    
    phases = ['early', 'late']
    phase_stats = {}
    
    for phase in phases:
        phase_df = df[df['training_phase'] == phase]
        if len(phase_df) == 0:
            continue
            
        successful = phase_df[phase_df['is_successful']]
        unsuccessful = phase_df[~phase_df['is_successful']]
        
        phase_stats[phase] = {
            'total': len(phase_df),
            'successful': len(successful),
            'unsuccessful': len(unsuccessful),
            'success_rate': len(successful) / len(phase_df) if len(phase_df) > 0 else 0,
            'avg_steps_successful': successful['num_steps'].mean() if len(successful) > 0 else 0,
            'avg_steps_unsuccessful': unsuccessful['num_steps'].mean() if len(unsuccessful) > 0 else 0,
            'avg_steps_all': phase_df['num_steps'].mean(),
            'median_steps_successful': successful['num_steps'].median() if len(successful) > 0 else 0,
            'median_steps_unsuccessful': unsuccessful['num_steps'].median() if len(unsuccessful) > 0 else 0,
            'avg_success_score': phase_df['success_score'].mean()
        }
        
        print(f"\n{phase.upper()} Training (First 20% vs Last 20%):")
        print(f"  Total rollouts: {phase_stats[phase]['total']}")
        print(f"  Successful: {phase_stats[phase]['successful']} ({phase_stats[phase]['success_rate']:.3f})")
        print(f"  Unsuccessful: {phase_stats[phase]['unsuccessful']}")
        print(f"  Average steps (successful): {phase_stats[phase]['avg_steps_successful']:.1f}")
        print(f"  Average steps (unsuccessful): {phase_stats[phase]['avg_steps_unsuccessful']:.1f}")
        print(f"  Median steps (successful): {phase_stats[phase]['median_steps_successful']:.1f}")
        print(f"  Median steps (unsuccessful): {phase_stats[phase]['median_steps_unsuccessful']:.1f}")
        print(f"  Average success score: {phase_stats[phase]['avg_success_score']:.3f}")
    
    # Compare phases
    if 'early' in phase_stats and 'late' in phase_stats:
        print(f"\n{'='*80}")
        print("PHASE COMPARISON")
        print(f"{'='*80}")
        
        early = phase_stats['early']
        late = phase_stats['late']
        
        print(f"\nSuccess rate improvement: {late['success_rate'] - early['success_rate']:.3f}")
        print(f"Average steps change (successful): {late['avg_steps_successful'] - early['avg_steps_successful']:.1f}")
        print(f"Average steps change (unsuccessful): {late['avg_steps_unsuccessful'] - early['avg_steps_unsuccessful']:.1f}")
        print(f"Success score improvement: {late['avg_success_score'] - early['avg_success_score']:.3f}")
    
    # Detailed distributions
    print(f"\n{'='*80}")
    print("STEP DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    
    for phase in phases:
        phase_df = df[df['training_phase'] == phase]
        if len(phase_df) == 0:
            continue
            
        print(f"\n{phase.upper()} Training:")
        
        # Step distribution for successful attempts
        successful = phase_df[phase_df['is_successful']]
        if len(successful) > 0:
            print(f"  Successful attempts ({len(successful)} rollouts):")
            print(f"    Steps: min={successful['num_steps'].min()}, max={successful['num_steps'].max()}")
            print(f"    Quartiles: Q1={successful['num_steps'].quantile(0.25):.1f}, "
                  f"Q2={successful['num_steps'].median():.1f}, Q3={successful['num_steps'].quantile(0.75):.1f}")
        
        # Step distribution for unsuccessful attempts
        unsuccessful = phase_df[~phase_df['is_successful']]
        if len(unsuccessful) > 0:
            print(f"  Unsuccessful attempts ({len(unsuccessful)} rollouts):")
            print(f"    Steps: min={unsuccessful['num_steps'].min()}, max={unsuccessful['num_steps'].max()}")
            print(f"    Quartiles: Q1={unsuccessful['num_steps'].quantile(0.25):.1f}, "
                  f"Q2={unsuccessful['num_steps'].median():.1f}, Q3={unsuccessful['num_steps'].quantile(0.75):.1f}")
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Termination Steps Analysis: Early vs Late Training (First 20% vs Last 20%)', fontsize=16, fontweight='bold')
    
    # 1. Success score distribution (left)
    ax1 = axes[0]
    early_scores = df[df['training_phase'] == 'early']['success_score']
    late_scores = df[df['training_phase'] == 'late']['success_score']
    
    if len(early_scores) > 0 and len(late_scores) > 0:
        # Handle the score=0 peak better by using log scale or separate treatment
        # Count zeros separately
        early_zeros = (early_scores == 0).sum()
        late_zeros = (late_scores == 0).sum()
        
        # Plot non-zero scores
        early_nonzero = early_scores[early_scores > 0]
        late_nonzero = late_scores[late_scores > 0]
        
        if len(early_nonzero) > 0 and len(late_nonzero) > 0:
            bins = np.linspace(0.01, 1, 40)
            ax1.hist(early_nonzero, bins=bins, alpha=0.7, label='Early Training (First 20%)', 
                    color='skyblue', density=False)
            ax1.hist(late_nonzero, bins=bins, alpha=0.7, label='Late Training (Last 20%)', 
                    color='orange', density=False)
        
        ax1.axvline(x=0.3, color='red', linestyle='--', label='Success Threshold (0.3)', linewidth=2)
        ax1.set_title('Non-Zero Score Distribution')
        ax1.set_xlabel('Success Score')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.01, 1)
        
        # Set x-axis ticks to exclude 0.0
        ax1.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # 2. Step distribution by success/failure and early/late (right)
    ax2 = axes[1]
    
    # Prepare data for violin plots: early_success, early_fail, late_success, late_fail
    violin_data = []
    violin_labels = []
    violin_colors = []
    
    for phase in ['early', 'late']:
        phase_df = df[df['training_phase'] == phase]
        if len(phase_df) == 0:
            continue
            
        successful = phase_df[phase_df['is_successful']]['num_steps']
        unsuccessful = phase_df[~phase_df['is_successful']]['num_steps']
        
        if len(successful) > 0:
            violin_data.append(successful)
            violin_labels.append(f'{phase.title()}\nSuccessful')
            violin_colors.append('lightgreen' if phase == 'early' else 'darkgreen')
        
        if len(unsuccessful) > 0:
            violin_data.append(unsuccessful)
            violin_labels.append(f'{phase.title()}\nUnsuccessful')
            violin_colors.append('lightcoral' if phase == 'early' else 'darkred')
    
    if violin_data:
        positions = list(range(1, len(violin_data) + 1))
        parts = ax2.violinplot(violin_data, positions=positions, showmeans=False, showmedians=False)
        
        # Color the violin plots
        for pc, color in zip(parts['bodies'], violin_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Remove the blue bars (cbars, cmins, cmaxes)
        for partname in ('cbars', 'cmins', 'cmaxes'):
            if partname in parts:
                parts[partname].set_visible(False)
        
        ax2.set_xticks(positions)
        ax2.set_xticklabels(violin_labels)
        ax2.set_title('Step Distribution by Success/Failure and Training Phase')
        ax2.set_ylabel('Number of Steps')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return df

def save_results(df, run_id=None):
    """Save plot and data with standardized naming."""
    # Save the plot
    plot_path = get_output_filename('termination_analysis', run_id) + '.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.show()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze termination steps for successful vs unsuccessful attempts')
    parser.add_argument('--run-id', type=str, default=RUN_ID, 
                        help=f'WandB run ID (default: {RUN_ID})')
    args = parser.parse_args()
    
    # Get run
    run = get_run(ENTITY, PROJECT, args.run_id)
    print(f"Analyzing run: {run.name} (ID: {run.id})")
    
    # Extract all training tables
    tables = extract_all_training_tables(run)
    
    if not tables:
        print("No tables found!")
        return
    
    print(f"\nLoaded {len(tables)} tables")
    
    # Analyze termination patterns
    df = analyze_termination_patterns(tables)
    
    # Save results
    save_results(df, args.run_id)
    
    return df

if __name__ == "__main__":
    main()