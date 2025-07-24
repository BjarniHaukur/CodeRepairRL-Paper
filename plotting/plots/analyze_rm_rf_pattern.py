#!/usr/bin/env python3
"""
Analyze instances where the agent performs 'rm -rf' followed by 'git clone'
to exploit the reward system by getting marginal rewards from the diff between
BUGGY_COMMIT and master.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import wandb
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

from wandb_utils import get_run, get_history, extract_shell_commands, SHELL_PATTERN
from plot_config import ENTITY, PROJECT, RUN_ID, get_output_filename


def find_rm_rf_patterns(completion_text, table_info=None):
    """
    Find instances of 'rm -rf' commands, especially those followed by git clone.
    Returns detailed information about each instance.
    """
    patterns = []
    
    # Find all shell commands using the exact pattern from wandb_utils
    shell_matches = [(m.start(), m.group(1)) for m in SHELL_PATTERN.finditer(completion_text)]
    
    for i, (start_pos, command) in enumerate(shell_matches):
        command = command.strip()
        
        # Check for rm -rf patterns (various forms)
        if any(pattern in command for pattern in ['rm -rf', 'rm -fr', 'rm -r -f', 'rm -f -r']):
            pattern_info = {
                'command': command,
                'position': start_pos,
                'command_index': i,
                'followed_by_clone': False,
                'next_commands': [],
                'table_info': table_info
            }
            
            # Check next few commands for git clone
            for j in range(i + 1, min(i + 5, len(shell_matches))):
                next_command = shell_matches[j][1].strip()
                pattern_info['next_commands'].append(next_command)
                
                if 'git clone' in next_command:
                    pattern_info['followed_by_clone'] = True
                    pattern_info['clone_command'] = next_command
                    break
            
            # Extract context around the rm -rf command
            context_start = max(0, start_pos - 200)
            context_end = min(len(completion_text), start_pos + 500)
            pattern_info['context'] = completion_text[context_start:context_end]
            
            patterns.append(pattern_info)
    
    return patterns


def analyze_all_runs():
    """
    Analyze all runs in the project to find rm -rf patterns.
    Focus on the longest runs as they likely have more training.
    """
    print("="*80)
    print("Analyzing rm -rf Pattern Across All Runs")
    print("="*80)
    
    # Get all runs from the project
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    # Sort runs by duration/steps to focus on longer runs
    run_infos = []
    for run in tqdm(runs, desc="Collecting run information"):
        if run.state == "finished":
            try:
                duration = (run.summary.get('_runtime', 0) or 0)
                steps = (run.summary.get('_step', 0) or 0)
                run_infos.append({
                    'run': run,
                    'id': run.id,
                    'name': run.name,
                    'duration': duration,
                    'steps': steps,
                    'created': run.created_at
                })
            except:
                continue
    
    # Sort by steps (most trained runs first)
    run_infos.sort(key=lambda x: x['steps'], reverse=True)
    
    print(f"\nFound {len(run_infos)} finished runs")
    print(f"Analyzing top 10 longest runs...")
    
    all_patterns = []
    
    # Analyze top runs
    for run_info in run_infos[:10]:  # Top 10 longest runs
        run = run_info['run']
        print(f"\n{'='*60}")
        print(f"Analyzing run: {run.name} (ID: {run.id})")
        print(f"Steps: {run_info['steps']}, Duration: {run_info['duration']:.0f}s")
        
        # Get history
        try:
            history = get_history(run)
            
            # Find tables with completion data
            tables_with_data = []
            for i, row in enumerate(history):
                if any(col for col in row.keys() if col.startswith('CodeTraining/rollouts')):
                    tables_with_data.append((i, row))
            
            print(f"Found {len(tables_with_data)} training tables")
            
            run_patterns = []
            
            # Process each table
            for table_idx, (hist_idx, row) in enumerate(tqdm(tables_with_data[:50], desc="Processing tables")):  # Sample first 50
                for key, value in row.items():
                    if key.startswith('CodeTraining/rollouts') and isinstance(value, wandb.data_types.Table):
                        try:
                            df = value.get_dataframe()
                            if 'Completion' in df.columns:
                                # Process each rollout
                                for rollout_idx, rollout in df.iterrows():
                                    completion = rollout.get('Completion', '')
                                    if not completion:
                                        continue
                                    
                                    table_info = {
                                        'run_id': run.id,
                                        'run_name': run.name,
                                        'table_index': table_idx,
                                        'history_index': hist_idx,
                                        'rollout_index': rollout_idx,
                                        'global_step': row.get('_step', hist_idx),
                                        'reward': rollout.get('Unified_diff_similarity_reward_func', 0)
                                    }
                                    
                                    patterns = find_rm_rf_patterns(completion, table_info)
                                    if patterns:
                                        run_patterns.extend(patterns)
                        except Exception as e:
                            continue
            
            if run_patterns:
                print(f"\nFound {len(run_patterns)} rm -rf instances in this run")
                all_patterns.extend(run_patterns)
                
                # Show some examples
                print("\nExample patterns:")
                for i, pattern in enumerate(run_patterns[:3]):
                    print(f"\n--- Example {i+1} ---")
                    print(f"Command: {pattern['command']}")
                    print(f"Followed by clone: {pattern['followed_by_clone']}")
                    if pattern['followed_by_clone']:
                        print(f"Clone command: {pattern['clone_command']}")
                    print(f"Reward: {pattern['table_info']['reward']:.3f}")
        
        except Exception as e:
            print(f"Error processing run: {e}")
            continue
    
    return all_patterns


def generate_report(patterns):
    """
    Generate a detailed report of the rm -rf exploitation pattern.
    """
    print("\n" + "="*80)
    print("RM -RF EXPLOITATION PATTERN REPORT")
    print("="*80)
    
    if not patterns:
        print("No rm -rf patterns found!")
        return
    
    # Overall statistics
    total_patterns = len(patterns)
    with_clone = sum(1 for p in patterns if p['followed_by_clone'])
    
    print(f"\nTotal rm -rf instances found: {total_patterns}")
    print(f"Instances followed by git clone: {with_clone} ({with_clone/total_patterns*100:.1f}%)")
    
    # Group by run
    by_run = defaultdict(list)
    for pattern in patterns:
        run_id = pattern['table_info']['run_id']
        by_run[run_id].append(pattern)
    
    print(f"\nPattern distribution across {len(by_run)} runs:")
    for run_id, run_patterns in by_run.items():
        run_name = run_patterns[0]['table_info']['run_name']
        clone_count = sum(1 for p in run_patterns if p['followed_by_clone'])
        avg_reward = sum(p['table_info']['reward'] for p in run_patterns) / len(run_patterns)
        print(f"  {run_name} ({run_id}): {len(run_patterns)} instances, "
              f"{clone_count} with clone, avg reward: {avg_reward:.3f}")
    
    # Analyze rewards
    print("\n" + "="*60)
    print("REWARD ANALYSIS")
    print("="*60)
    
    rewards_with_clone = [p['table_info']['reward'] for p in patterns if p['followed_by_clone']]
    rewards_without_clone = [p['table_info']['reward'] for p in patterns if not p['followed_by_clone']]
    
    if rewards_with_clone:
        print(f"\nRewards for rm -rf followed by clone:")
        print(f"  Count: {len(rewards_with_clone)}")
        print(f"  Mean: {sum(rewards_with_clone)/len(rewards_with_clone):.3f}")
        print(f"  Max: {max(rewards_with_clone):.3f}")
        print(f"  Min: {min(rewards_with_clone):.3f}")
    
    if rewards_without_clone:
        print(f"\nRewards for rm -rf NOT followed by clone:")
        print(f"  Count: {len(rewards_without_clone)}")
        print(f"  Mean: {sum(rewards_without_clone)/len(rewards_without_clone):.3f}")
        print(f"  Max: {max(rewards_without_clone):.3f}")
        print(f"  Min: {min(rewards_without_clone):.3f}")
    
    # Show detailed examples
    print("\n" + "="*60)
    print("DETAILED EXAMPLES OF EXPLOITATION")
    print("="*60)
    
    exploitation_examples = [p for p in patterns if p['followed_by_clone'] and p['table_info']['reward'] > 0.1]
    exploitation_examples.sort(key=lambda x: x['table_info']['reward'], reverse=True)
    
    for i, pattern in enumerate(exploitation_examples[:5]):
        print(f"\n--- High-Reward Example {i+1} ---")
        print(f"Run: {pattern['table_info']['run_name']}")
        print(f"Step: {pattern['table_info']['global_step']}")
        print(f"Reward: {pattern['table_info']['reward']:.3f}")
        print(f"rm -rf command: {pattern['command']}")
        print(f"git clone command: {pattern.get('clone_command', 'N/A')}")
        print(f"\nContext snippet:")
        context = pattern['context'].replace('\n', ' ')[:300] + "..."
        print(f"  {context}")
    
    # Save detailed data
    import json
    output_file = get_output_filename('rm_rf_exploitation_analysis', 'all_runs') + '.json'
    with open(output_file, 'w') as f:
        # Convert patterns to serializable format
        serializable_patterns = []
        for p in patterns:
            sp = p.copy()
            sp['table_info'] = p['table_info'].copy()
            serializable_patterns.append(sp)
        
        json.dump({
            'total_patterns': total_patterns,
            'patterns_with_clone': with_clone,
            'runs_analyzed': len(by_run),
            'patterns': serializable_patterns
        }, f, indent=2)
    
    print(f"\nDetailed data saved to: {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze rm -rf exploitation pattern across runs')
    parser.add_argument('--max-runs', type=int, default=10, 
                        help='Maximum number of runs to analyze (default: 10)')
    args = parser.parse_args()
    
    # Analyze patterns across all runs
    patterns = analyze_all_runs()
    
    # Generate report
    generate_report(patterns)


if __name__ == "__main__":
    main()