#!/usr/bin/env python3
"""
Analyze how individual coding problems perform across training epochs.
Tracks reward distribution for each unique problem over time to understand
if improved average rewards come from a subset of "easy" problems or uniform improvement.
"""


import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import tempfile
import re
from typing import Dict, List, Tuple, Optional
import json

from plot_config import (
    ENTITY, PROJECT, RUN_ID, FILENAME_FORMAT,
    setup_plotting_style, create_figure, save_figure, format_axis_labels,
    COLORS, get_color, get_output_filename
)
from wandb_utils import get_run, get_history
from utils.table_parser import TableExtractor


def get_problem_key(prompt: str) -> str:
    """Get a unique key for the problem from the user prompt."""
    # Split on "<|im_start|>user" and take the last part (the actual user prompt)
    parts = prompt.split("<|im_start|>user")
    if len(parts) > 1:
        return parts[-1].strip()
    return prompt.strip()


def calculate_gini(values):
    """Calculate Gini coefficient to measure inequality in rewards."""
    values = np.array(values)
    if len(values) == 0:
        return 0
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0



def process_single_table(run, table_index, row_data):
    """Process a single table - used for parallel processing."""
    global_step = int(row_data['train/global_step'])
    table_info = row_data['table']
    table_path = table_info['path']
    
    try:
        # Download and parse table (use replace=False to cache)
        table_file = run.file(table_path)
        html_content = table_file.download(replace=False, exist_ok=True)
        
        with open(html_content.name, 'r', encoding='utf-8') as f:
            html_data = f.read()
        
        soup = BeautifulSoup(html_data, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return None
        
        # Extract headers
        headers = []
        header_row = table.find('thead')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
        else:
            first_row = table.find('tr')
            if first_row:
                headers = [td.get_text(strip=True) for td in first_row.find_all(['td', 'th'])]
        
        # Extract rows
        rows = []
        tbody = table.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                row = [td.get_text(separator='\n', strip=True) for td in tr.find_all('td')]
                if row:
                    rows.append(row)
        
        if headers and rows:
            # Create DataFrame
            max_cols = len(headers)
            padded_rows = [row[:max_cols] + [''] * (max_cols - len(row)) for row in rows]
            
            df = pd.DataFrame(padded_rows, columns=headers)
            df['global_step'] = global_step
            df['table_index'] = table_index
            return df
            
    except Exception as e:
        return None


def extract_all_training_tables(run, max_tables=None):
    """Extract all tables from the training history using parallel processing."""
    print("="*60)
    print("Extracting All Training Tables (Parallel)")
    print("="*60)
    
    history = get_history(run, keys=['train/global_step', 'table'])
    table_history = history[~history['table'].isna()]
    
    print(f"Found {len(table_history)} table entries in training history")
    
    if max_tables:
        table_history = table_history.head(max_tables)
        print(f"Limited to {len(table_history)} tables for testing")
    
    # Prepare data for parallel processing
    table_data = [(i, row) for i, (_, row) in enumerate(table_history.iterrows())]
    
    tables = []
    failed_count = 0
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_table, run, i, row): i 
            for i, row in table_data
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(table_data), desc="Processing tables") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result is not None:
                        tables.append((index, result))
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                pbar.update(1)
    
    # Sort tables by index to maintain order
    tables.sort(key=lambda x: x[0])
    tables = [df for _, df in tables]
    
    print(f"\nSuccessfully extracted {len(tables)} tables ({failed_count} failed)")
    return tables


def analyze_individual_problems(run_id: str, max_tables: Optional[int] = None) -> Dict:
    """
    Analyze how individual problems perform across all training tables.
    
    Args:
        run_id: W&B run ID
        max_tables: Maximum number of tables to process (None = all)
    
    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing individual problem performance for run: {run_id}")
    
    run = get_run(ENTITY, PROJECT, run_id)
    
    # Extract all training tables
    # Use new TableExtractor utility
    extractor = TableExtractor()
    tables = extractor.extract_all_training_tables(run, max_tables=max_tables)
    
    if not tables:
        print("No tables found!")
        return {
            'problem_data': {},
            'problem_stats': {},
            'global_stats': {},
            'num_problems': 0,
            'num_tables': 0,
            'total_samples': 0
        }
    
    print(f"Processing {len(tables)} tables...")
    
    # Store all data points: problem_key -> list of (global_step, reward) tuples
    problem_data = defaultdict(list)
    all_rewards_by_step = defaultdict(list)  # global_step -> list of rewards
    total_samples = 0
    
    for table_df in tables:
        global_step = table_df['global_step'].iloc[0]
        table_rewards = []
        
        # Process each rollout in this table
        for _, row in table_df.iterrows():
            # Extract prompt and rewards
            prompt = row.get('Prompt', '')
            
            # Try different reward column names
            reward = None
            reward_columns = [
                'Unified_diff_similarity_reward_func',
                'Unified_diff_file_match_reward_func',
                'Reward',
                'reward'
            ]
            
            for col in reward_columns:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    try:
                        reward = float(row[col])
                        break
                    except (ValueError, TypeError):
                        continue
            
            if prompt and reward is not None:
                problem_key = get_problem_key(prompt)
                problem_data[problem_key].append((global_step, reward))
                all_rewards_by_step[global_step].append(reward)
                table_rewards.append(reward)
                total_samples += 1
        
        if table_rewards:
            print(f"Global step {global_step}: {len(table_rewards)} samples, mean reward: {np.mean(table_rewards):.3f}")
    
    # Calculate per-problem statistics
    problem_stats = {}
    for problem_key, data_points in problem_data.items():
        if len(data_points) >= 1:  # At least one data point
            steps = [d[0] for d in data_points]
            rewards = [d[1] for d in data_points]
            
            problem_stats[problem_key] = {
                'num_samples': len(data_points),
                'first_step': int(min(steps)),
                'last_step': int(max(steps)),
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'min_reward': float(np.min(rewards)),
                'max_reward': float(np.max(rewards)),
                'data_points': data_points
            }
    
    # Calculate global statistics
    global_stats = {}
    if all_rewards_by_step:
        all_rewards = [r for rewards in all_rewards_by_step.values() for r in rewards]
        global_stats = {
            'overall_mean': float(np.mean(all_rewards)),
            'overall_std': float(np.std(all_rewards)),
            'overall_min': float(np.min(all_rewards)),
            'overall_max': float(np.max(all_rewards)),
            'num_steps': len(all_rewards_by_step),
            'rewards_by_step': {str(k): v for k, v in all_rewards_by_step.items()}  # Convert keys to strings
        }
    
    print(f"\nSummary:")
    print(f"  Unique problems: {len(problem_stats)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Tables processed: {len(tables)}")
    if global_stats:
        print(f"  Overall mean reward: {global_stats['overall_mean']:.3f}")
    
    return {
        'problem_data': dict(problem_data),
        'problem_stats': problem_stats,
        'global_stats': global_stats,
        'num_problems': len(problem_stats),
        'num_tables': len(tables),
        'total_samples': total_samples
    }


def analyze_problem_insights(analysis: Dict):
    """Extract deep insights about problem difficulty and patterns."""
    problem_stats = analysis['problem_stats']
    
    if not problem_stats:
        return {}
    
    # Convert to list for easier sorting
    problems_list = [(key, stats) for key, stats in problem_stats.items()]
    
    # Sort by mean reward
    problems_by_reward = sorted(problems_list, key=lambda x: x[1]['mean_reward'])
    
    # Calculate percentiles
    n_problems = len(problems_list)
    top_10_pct_idx = int(n_problems * 0.9)
    bottom_10_pct_idx = int(n_problems * 0.1)
    
    insights = {
        'total_problems': n_problems,
        'hardest_10pct': problems_by_reward[:bottom_10_pct_idx],
        'easiest_10pct': problems_by_reward[top_10_pct_idx:],
        'zero_reward_problems': [p for p in problems_list if p[1]['mean_reward'] == 0.0],
        'high_variance_problems': sorted(problems_list, key=lambda x: x[1]['std_reward'], reverse=True)[:10],
        'most_frequent_problems': sorted(problems_list, key=lambda x: x[1]['num_samples'], reverse=True)[:10],
        'least_frequent_problems': sorted(problems_list, key=lambda x: x[1]['num_samples'])[:10],
    }
    
    # Analyze reward patterns
    reward_thresholds = {
        'zero': [],
        'very_low': [],  # 0 < r <= 0.05
        'low': [],       # 0.05 < r <= 0.1
        'medium': [],    # 0.1 < r <= 0.3
        'high': [],      # 0.3 < r <= 0.5
        'very_high': []  # r > 0.5
    }
    
    for key, stats in problem_stats.items():
        mean_r = stats['mean_reward']
        if mean_r == 0:
            reward_thresholds['zero'].append((key, stats))
        elif mean_r <= 0.05:
            reward_thresholds['very_low'].append((key, stats))
        elif mean_r <= 0.1:
            reward_thresholds['low'].append((key, stats))
        elif mean_r <= 0.3:
            reward_thresholds['medium'].append((key, stats))
        elif mean_r <= 0.5:
            reward_thresholds['high'].append((key, stats))
        else:
            reward_thresholds['very_high'].append((key, stats))
    
    insights['reward_distribution'] = {k: len(v) for k, v in reward_thresholds.items()}
    insights['reward_thresholds'] = reward_thresholds
    
    return insights


def analyze_problem_patterns(problem_stats: Dict) -> Dict:
    """Deep textual analysis of problem patterns and characteristics."""
    
    print("\n" + "="*100)
    print("DEEP TEXTUAL ANALYSIS OF INDIVIDUAL PROBLEM PERFORMANCE")
    print("="*100)
    
    if not problem_stats:
        print("No problem data available for analysis.")
        return {}
    
    # Convert to list for easier analysis
    problems_list = [(key, stats) for key, stats in problem_stats.items()]
    
    print(f"\n📊 DATASET OVERVIEW:")
    print("-" * 60)
    print(f"Total unique problems analyzed: {len(problems_list)}")
    
    # Calculate overall statistics
    all_means = [stats['mean_reward'] for _, stats in problems_list]
    all_stds = [stats['std_reward'] for _, stats in problems_list]
    all_samples = [stats['num_samples'] for _, stats in problems_list]
    
    print(f"Overall mean reward range: [{min(all_means):.4f}, {max(all_means):.4f}]")
    print(f"Overall std deviation range: [{min(all_stds):.4f}, {max(all_stds):.4f}]")
    print(f"Sample count range: [{min(all_samples)}, {max(all_samples)}]")
    print(f"Total samples across all problems: {sum(all_samples)}")
    
    # Sort by different criteria for analysis
    by_mean = sorted(problems_list, key=lambda x: x[1]['mean_reward'])
    by_std = sorted(problems_list, key=lambda x: x[1]['std_reward'], reverse=True)
    by_frequency = sorted(problems_list, key=lambda x: x[1]['num_samples'], reverse=True)
    
    # Find special categories
    always_zero = [(k, s) for k, s in problems_list if s['max_reward'] == 0.0]
    always_nonzero = [(k, s) for k, s in problems_list if s['min_reward'] > 0.0]
    perfect_problems = [(k, s) for k, s in problems_list if s['mean_reward'] == 1.0]
    zero_variance = [(k, s) for k, s in problems_list if s['std_reward'] == 0.0]
    
    print(f"\n🔍 SPECIAL CATEGORIES:")
    print("-" * 60)
    print(f"Problems ALWAYS getting 0.0 reward: {len(always_zero)}")
    print(f"Problems NEVER getting 0.0 reward: {len(always_nonzero)}")
    print(f"Problems with perfect 1.0 mean reward: {len(perfect_problems)}")
    print(f"Problems with zero variance (completely consistent): {len(zero_variance)}")
    
    # Detailed analysis of extreme cases
    print(f"\n🔴 HARDEST PROBLEMS (lowest mean rewards):")
    print("-" * 60)
    for i, (key, stats) in enumerate(by_mean[:10]):
        key_preview = key[:80] + "..." if len(key) > 80 else key
        print(f"{i+1:2d}. Mean: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f} "
              f"(n={stats['num_samples']:3d}) | {key_preview}")
    
    print(f"\n🟢 EASIEST PROBLEMS (highest mean rewards):")
    print("-" * 60)
    for i, (key, stats) in enumerate(by_mean[-10:]):
        key_preview = key[:80] + "..." if len(key) > 80 else key
        print(f"{i+1:2d}. Mean: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f} "
              f"(n={stats['num_samples']:3d}) | {key_preview}")
    
    print(f"\n📈 MOST INCONSISTENT PROBLEMS (highest std deviation):")
    print("-" * 60)
    for i, (key, stats) in enumerate(by_std[:10]):
        key_preview = key[:80] + "..." if len(key) > 80 else key
        print(f"{i+1:2d}. Std: {stats['std_reward']:.4f} | Mean: {stats['mean_reward']:.4f} "
              f"(n={stats['num_samples']:3d}) | {key_preview}")
    
    print(f"\n📌 MOST FREQUENT PROBLEMS:")
    print("-" * 60)
    for i, (key, stats) in enumerate(by_frequency[:10]):
        key_preview = key[:80] + "..." if len(key) > 80 else key
        print(f"{i+1:2d}. Samples: {stats['num_samples']:3d} | Mean: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f} | {key_preview}")
    
    # Analysis of always-zero problems
    if always_zero:
        print(f"\n❌ PROBLEMS THAT ALWAYS FAIL (mean=max=0.0):")
        print("-" * 60)
        for i, (key, stats) in enumerate(always_zero[:5]):
            key_preview = key[:80] + "..." if len(key) > 80 else key
            print(f"{i+1:2d}. Failed {stats['num_samples']:3d} times | {key_preview}")
        
        if len(always_zero) > 5:
            print(f"... and {len(always_zero) - 5} more problems that always fail")
    
    # Analysis of always-nonzero problems
    if always_nonzero:
        print(f"\n✅ PROBLEMS THAT NEVER FAIL (min>0.0):")
        print("-" * 60)
        for i, (key, stats) in enumerate(always_nonzero[:5]):
            key_preview = key[:80] + "..." if len(key) > 80 else key
            print(f"{i+1:2d}. Min: {stats['min_reward']:.4f} | Mean: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f} "
                  f"(n={stats['num_samples']:3d}) | {key_preview}")
        
        if len(always_nonzero) > 5:
            print(f"... and {len(always_nonzero) - 5} more problems that never fail")
    
    # Consistency analysis
    if zero_variance:
        print(f"\n🎯 PERFECTLY CONSISTENT PROBLEMS (std=0.0):")
        print("-" * 60)
        for i, (key, stats) in enumerate(zero_variance[:5]):
            key_preview = key[:80] + "..." if len(key) > 80 else key
            print(f"{i+1:2d}. Always: {stats['mean_reward']:.4f} "
                  f"(n={stats['num_samples']:3d}) | {key_preview}")
        
        if len(zero_variance) > 5:
            print(f"... and {len(zero_variance) - 5} more perfectly consistent problems")
    
    # Statistical insights
    print(f"\n📊 STATISTICAL INSIGHTS:")
    print("-" * 60)
    
    # Correlation analysis
    correlation_freq_mean = np.corrcoef(all_samples, all_means)[0, 1] if len(all_samples) > 1 else 0
    correlation_freq_std = np.corrcoef(all_samples, all_stds)[0, 1] if len(all_samples) > 1 else 0
    correlation_mean_std = np.corrcoef(all_means, all_stds)[0, 1] if len(all_means) > 1 else 0
    
    print(f"Correlation between frequency and mean reward: {correlation_freq_mean:.3f}")
    print(f"Correlation between frequency and std deviation: {correlation_freq_std:.3f}")
    print(f"Correlation between mean reward and std deviation: {correlation_mean_std:.3f}")
    
    # Distribution analysis
    reward_ranges = {
        'zero': len([s for _, s in problems_list if s['mean_reward'] == 0.0]),
        'very_low': len([s for _, s in problems_list if 0.0 < s['mean_reward'] <= 0.1]),
        'low': len([s for _, s in problems_list if 0.1 < s['mean_reward'] <= 0.3]),
        'medium': len([s for _, s in problems_list if 0.3 < s['mean_reward'] <= 0.6]),
        'high': len([s for _, s in problems_list if 0.6 < s['mean_reward'] <= 0.9]),
        'very_high': len([s for _, s in problems_list if s['mean_reward'] > 0.9])
    }
    
    print(f"\nReward distribution:")
    total = len(problems_list)
    for category, count in reward_ranges.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {category:10}: {count:3d} problems ({pct:5.1f}%)")
    
    # Frequency distribution
    freq_ranges = {
        'rare': len([s for _, s in problems_list if s['num_samples'] <= 5]),
        'uncommon': len([s for _, s in problems_list if 5 < s['num_samples'] <= 15]),
        'common': len([s for _, s in problems_list if 15 < s['num_samples'] <= 50]),
        'frequent': len([s for _, s in problems_list if s['num_samples'] > 50])
    }
    
    print(f"\nFrequency distribution:")
    for category, count in freq_ranges.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {category:10}: {count:3d} problems ({pct:5.1f}%)")
    
    print(f"\n🔍 KEY FINDINGS:")
    print("-" * 60)
    
    # Calculate some key insights
    median_mean = np.median(all_means)
    q1_mean = np.percentile(all_means, 25)
    q3_mean = np.percentile(all_means, 75)
    
    print(f"• Median problem difficulty (mean reward): {median_mean:.4f}")
    print(f"• 25% of problems have mean reward ≤ {q1_mean:.4f}")
    print(f"• 75% of problems have mean reward ≤ {q3_mean:.4f}")
    print(f"• Most frequent problem appears {max(all_samples)} times")
    print(f"• {len(always_zero)} problems ({len(always_zero)/total*100:.1f}%) never succeed")
    print(f"• {len(always_nonzero)} problems ({len(always_nonzero)/total*100:.1f}%) never fail")
    print(f"• {len(zero_variance)} problems ({len(zero_variance)/total*100:.1f}%) are perfectly consistent")
    
    if correlation_freq_mean > 0.3:
        print(f"• Strong positive correlation: frequent problems tend to be easier")
    elif correlation_freq_mean < -0.3:
        print(f"• Strong negative correlation: frequent problems tend to be harder")
    else:
        print(f"• Weak correlation between frequency and difficulty")
    
    return {
        'always_zero': always_zero,
        'always_nonzero': always_nonzero,
        'perfect_problems': perfect_problems,
        'zero_variance': zero_variance,
        'correlations': {
            'freq_mean': correlation_freq_mean,
            'freq_std': correlation_freq_std,
            'mean_std': correlation_mean_std
        },
        'distributions': {
            'reward_ranges': reward_ranges,
            'freq_ranges': freq_ranges
        }
    }


def analyze_learning_progression(problem_stats: Dict) -> Dict:
    """Analyze whether problems improve from first 50% to latter 50% of rollouts."""
    
    improved_count = 0
    declined_count = 0
    unchanged_count = 0
    total_analyzable = 0
    
    improved_problems = []  # (problem_key, improvement_amount)
    declined_problems = []  # (problem_key, decline_amount)
    
    for problem_key, stats in problem_stats.items():
        data_points = stats['data_points']  # List of (global_step, reward) tuples
        
        # Only analyze problems with sufficient data (at least 4 samples)
        if len(data_points) < 4:
            continue
        
        total_analyzable += 1
        
        # Sort by global step to get chronological order
        sorted_data = sorted(data_points, key=lambda x: x[0])
        
        # Split into first and latter halves
        mid_point = len(sorted_data) // 2
        first_half = sorted_data[:mid_point]
        latter_half = sorted_data[mid_point:]
        
        # Calculate average rewards for each half
        first_half_avg = np.mean([reward for _, reward in first_half])
        latter_half_avg = np.mean([reward for _, reward in latter_half])
        
        improvement = latter_half_avg - first_half_avg
        
        if latter_half_avg > first_half_avg:
            improved_count += 1
            improved_problems.append((problem_key, improvement))
        elif latter_half_avg < first_half_avg:
            declined_count += 1
            declined_problems.append((problem_key, improvement))  # improvement will be negative
        else:
            unchanged_count += 1
    
    # Sort by improvement/decline amount
    improved_problems.sort(key=lambda x: x[1], reverse=True)  # Largest improvements first
    declined_problems.sort(key=lambda x: x[1])  # Largest declines first (most negative)
    
    improvement_percentage = (improved_count / total_analyzable * 100) if total_analyzable > 0 else 0
    
    return {
        'improved_count': improved_count,
        'declined_count': declined_count,
        'unchanged_count': unchanged_count,
        'total_analyzable': total_analyzable,
        'improvement_percentage': improvement_percentage,
        'improved_problems': improved_problems,
        'declined_problems': declined_problems
    }


def plot_problem_insights(analysis: Dict, insights: Dict, run_id: str):
    """Create comprehensive visualizations of problem insights."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    problem_stats = analysis['problem_stats']
    
    # 1. Reward distribution histogram
    ax1 = fig.add_subplot(gs[0, 0])
    mean_rewards = [stats['mean_reward'] for stats in problem_stats.values()]
    ax1.hist(mean_rewards, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.axvline(x=np.mean(mean_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(mean_rewards):.3f}')
    ax1.set_xlabel('Mean Reward')
    ax1.set_ylabel('Number of Problems')
    ax1.set_title('Distribution of Problem Difficulty')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Variance vs Mean Reward
    ax2 = fig.add_subplot(gs[0, 1])
    std_rewards = [stats['std_reward'] for stats in problem_stats.values()]
    ax2.scatter(mean_rewards, std_rewards, alpha=0.6, s=30)
    ax2.set_xlabel('Mean Reward')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Reward Consistency vs Difficulty')
    ax2.grid(True, alpha=0.3)
    
    # 3. Frequency vs Difficulty
    ax3 = fig.add_subplot(gs[0, 2])
    sample_counts = [stats['num_samples'] for stats in problem_stats.values()]
    ax3.scatter(sample_counts, mean_rewards, alpha=0.6, s=30, c=std_rewards, cmap='viridis')
    ax3.set_xlabel('Number of Appearances')
    ax3.set_ylabel('Mean Reward')
    ax3.set_title('Problem Frequency vs Difficulty')
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Std Dev')
    ax3.grid(True, alpha=0.3)
    
    # 4. Reward categories pie chart
    ax4 = fig.add_subplot(gs[1, 0])
    dist = insights['reward_distribution']
    labels = [f"{k}\n({v} problems)" for k, v in dist.items() if v > 0]
    sizes = [v for v in dist.values() if v > 0]
    colors = ['#ff4444', '#ff8844', '#ffaa44', '#44ff44', '#44ff88', '#44ffaa']
    ax4.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(sizes)])
    ax4.set_title('Problem Difficulty Categories')
    
    # 5. Top/Bottom performers
    ax5 = fig.add_subplot(gs[1, 1:])
    top_5 = insights['easiest_10pct'][:5]
    bottom_5 = insights['hardest_10pct'][:5]
    
    y_pos = np.arange(10)
    rewards = ([stats['mean_reward'] for _, stats in bottom_5] + 
               [stats['mean_reward'] for _, stats in top_5])
    labels = ([f"Hard {i+1}" for i in range(5)] + 
              [f"Easy {i+1}" for i in range(5)])
    colors = ['#ff4444'] * 5 + ['#44ff44'] * 5
    
    bars = ax5.barh(y_pos, rewards, color=colors, alpha=0.7)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(labels)
    ax5.set_xlabel('Mean Reward')
    ax5.set_title('Hardest vs Easiest Problems')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, reward) in enumerate(zip(bars, rewards)):
        ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{reward:.3f}', va='center')
    
    # 6. Temporal analysis - when do problems appear
    ax6 = fig.add_subplot(gs[2, :])
    all_steps = []
    all_rewards = []
    for stats in problem_stats.values():
        for step, reward in stats['data_points']:
            all_steps.append(step)
            all_rewards.append(reward)
    
    # Create 2D histogram
    h = ax6.hist2d(all_steps, all_rewards, bins=[30, 20], cmap='YlOrRd', alpha=0.8)
    plt.colorbar(h[3], ax=ax6, label='Count')
    ax6.set_xlabel('Training Step')
    ax6.set_ylabel('Reward')
    ax6.set_title('Reward Distribution Over Training Time')
    
    plt.suptitle(f'Deep Problem Analysis - Run {run_id}', fontsize=16)
    
    filename = FILENAME_FORMAT.format(name='problem_insights', run_id=run_id)
    save_figure(fig, filename)
    plt.close()


def analyze_temporal_patterns(analysis: Dict):
    """Analyze how problem difficulty changes over time."""
    problem_data = analysis['problem_data']
    global_stats = analysis['global_stats']
    
    # Group problems by when they first appear
    early_problems = []
    middle_problems = []
    late_problems = []
    
    if 'rewards_by_step' in global_stats:
        steps = sorted([int(s) for s in global_stats['rewards_by_step'].keys()])
        if steps:
            early_cutoff = steps[len(steps)//3]
            late_cutoff = steps[2*len(steps)//3]
            
            for problem_key, data_points in problem_data.items():
                first_step = min(dp[0] for dp in data_points)
                if first_step <= early_cutoff:
                    early_problems.append(problem_key)
                elif first_step >= late_cutoff:
                    late_problems.append(problem_key)
                else:
                    middle_problems.append(problem_key)
    
    print("\n⏱️  TEMPORAL PATTERNS:")
    print("-" * 40)
    print(f"Problems appearing early in training: {len(early_problems)}")
    print(f"Problems appearing mid training: {len(middle_problems)}")
    print(f"Problems appearing late in training: {len(late_problems)}")
    
    return {
        'early_problems': early_problems,
        'middle_problems': middle_problems,
        'late_problems': late_problems
    }

def save_analysis_results(analysis: Dict, output_path: str):
    """Save analysis results to JSON for later use."""
    # Convert to serializable format (excluding large data structures)
    serializable = {
        'num_problems': analysis['num_problems'],
        'num_tables': analysis['num_tables'],
        'total_samples': analysis['total_samples'],
        'global_stats': analysis['global_stats'],
        'problem_summary': {}
    }
    
    # Summarize problem stats (excluding full data points)
    for problem_key, stats in analysis['problem_stats'].items():
        serializable['problem_summary'][problem_key] = {
            'num_samples': stats['num_samples'],
            'first_step': stats['first_step'],
            'last_step': stats['last_step'],
            'mean_reward': stats['mean_reward'],
            'std_reward': stats['std_reward'],
            'min_reward': stats['min_reward'],
            'max_reward': stats['max_reward']
        }
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"Analysis results saved to: {output_path}")


def generate_prompt_analysis(learning_progression_insights: Dict, problem_stats: Dict, output_filename: str):
    """Generate detailed prompt analysis by category for manual review."""
    
    # Get categorized problems
    improved_problems = learning_progression_insights['improved_problems'][:10]
    declined_problems = learning_progression_insights['declined_problems'][:10]
    
    # Get always failing problems (from problem_stats)
    always_failing = [(k, s) for k, s in problem_stats.items() if s['max_reward'] == 0.0][:10]
    
    # Get never failing problems
    never_failing = [(k, s) for k, s in problem_stats.items() if s['min_reward'] > 0.0][:10]
    
    # Get most inconsistent problems
    most_inconsistent = sorted(problem_stats.items(), key=lambda x: x[1]['std_reward'], reverse=True)[:10]
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("DETAILED PROMPT ANALYSIS BY CATEGORY\n")
        f.write("="*80 + "\n\n")
        f.write("Instructions: For each prompt, add your analysis of difficulty and a heuristic score (1-5):\n")
        f.write("1 = Very Easy (simple syntax/format fixes)\n")
        f.write("2 = Easy (straightforward bug fixes, clear issues)\n") 
        f.write("3 = Medium (moderate complexity, some reasoning required)\n")
        f.write("4 = Hard (complex logic, multiple systems, edge cases)\n")
        f.write("5 = Very Hard (deep architectural issues, advanced concepts)\n\n")
        
        # Most Improved Problems
        f.write("CATEGORY 1: MOST IMPROVED PROBLEMS (latter 50% >> first 50%)\n")
        f.write("-" * 60 + "\n")
        f.write("These problems showed the largest improvement during training.\n\n")
        
        for i, (problem_key, improvement) in enumerate(improved_problems):
            f.write(f"PROMPT {i+1}: (Improvement: +{improvement:.4f})\n")
            f.write(f"{problem_key}\n")
            f.write("ANALYSIS: [Your difficulty analysis here]\n")
            f.write("DIFFICULTY: [1-5]\n\n")
        
        # Most Declined Problems  
        f.write("\nCATEGORY 2: MOST DECLINED PROBLEMS (latter 50% << first 50%)\n")
        f.write("-" * 60 + "\n")
        f.write("These problems showed the largest decline during training.\n\n")
        
        for i, (problem_key, decline) in enumerate(declined_problems):
            f.write(f"PROMPT {i+1}: (Decline: {decline:.4f})\n")
            f.write(f"{problem_key}\n")
            f.write("ANALYSIS: [Your difficulty analysis here]\n")
            f.write("DIFFICULTY: [1-5]\n\n")
        
        # Always Failing Problems
        f.write("\nCATEGORY 3: ALWAYS FAILING PROBLEMS (mean = 0.0)\n")
        f.write("-" * 60 + "\n")
        f.write("These problems never succeeded throughout training.\n\n")
        
        for i, (problem_key, stats) in enumerate(always_failing):
            f.write(f"PROMPT {i+1}: (Failed {stats['num_samples']} times)\n")
            f.write(f"{problem_key}\n")
            f.write("ANALYSIS: [Your difficulty analysis here]\n")
            f.write("DIFFICULTY: [1-5]\n\n")
        
        # Never Failing Problems
        f.write("\nCATEGORY 4: NEVER FAILING PROBLEMS (min > 0.0)\n")
        f.write("-" * 60 + "\n")
        f.write("These problems always achieved some reward.\n\n")
        
        for i, (problem_key, stats) in enumerate(never_failing):
            f.write(f"PROMPT {i+1}: (Min: {stats['min_reward']:.4f}, Mean: {stats['mean_reward']:.4f})\n")
            f.write(f"{problem_key}\n")
            f.write("ANALYSIS: [Your difficulty analysis here]\n")
            f.write("DIFFICULTY: [1-5]\n\n")
        
        # Most Inconsistent Problems
        f.write("\nCATEGORY 5: MOST INCONSISTENT PROBLEMS (highest std dev)\n")
        f.write("-" * 60 + "\n")
        f.write("These problems showed the highest variance in performance.\n\n")
        
        for i, (problem_key, stats) in enumerate(most_inconsistent):
            f.write(f"PROMPT {i+1}: (Mean: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f})\n")
            f.write(f"{problem_key}\n")
            f.write("ANALYSIS: [Your difficulty analysis here]\n")
            f.write("DIFFICULTY: [1-5]\n\n")
    
    print(f"Detailed prompt analysis saved to: {output_filename}")


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze individual problem performance across all training')
    parser.add_argument('--run-id', type=str, default=RUN_ID,
                        help=f'W&B run ID to analyze (default: {RUN_ID})')
    parser.add_argument('--max-tables', type=int, default=None,
                        help='Maximum number of tables to process (default: all)')
    args = parser.parse_args()
    
    # Use provided run ID
    run_id = args.run_id
    max_tables = args.max_tables if args.max_tables and args.max_tables > 0 else None
    
    print("="*80)
    print("INDIVIDUAL PROBLEM PERFORMANCE ANALYSIS")
    print(f"Run ID: {run_id}")
    print("="*80)
    
    # Run analysis
    analysis = analyze_individual_problems(run_id, max_tables=max_tables)
    
    print(f"\nAnalysis complete!")
    print(f"  Unique problems: {analysis['num_problems']}")
    print(f"  Total samples: {analysis['total_samples']}")
    print(f"  Tables processed: {analysis['num_tables']}")
    
    # Deep textual analysis (replacing plots)
    textual_insights = analyze_problem_patterns(analysis['problem_stats'])
    
    # Analyze learning progression per problem
    learning_progression_insights = analyze_learning_progression(analysis['problem_stats'])
    print(f"\n🚀 LEARNING PROGRESSION ANALYSIS:")
    print("-" * 60)
    print(f"Percentage of problems where latter 50% > first 50%: {learning_progression_insights['improvement_percentage']:.1f}%")
    print(f"Problems showing improvement: {learning_progression_insights['improved_count']}/{learning_progression_insights['total_analyzable']}")
    print(f"Problems showing decline: {learning_progression_insights['declined_count']}/{learning_progression_insights['total_analyzable']}")
    print(f"Problems with no change: {learning_progression_insights['unchanged_count']}/{learning_progression_insights['total_analyzable']}")
    
    if learning_progression_insights['improved_problems']:
        print(f"\nTop 5 most improved problems:")
        for i, (key, improvement) in enumerate(learning_progression_insights['improved_problems'][:5]):
            key_preview = key[:80] + "..." if len(key) > 80 else key
            print(f"  {i+1}. Improvement: +{improvement:.4f} | {key_preview}")
    
    if learning_progression_insights['declined_problems']:
        print(f"\nTop 5 most declined problems:")
        for i, (key, decline) in enumerate(learning_progression_insights['declined_problems'][:5]):
            key_preview = key[:80] + "..." if len(key) > 80 else key
            print(f"  {i+1}. Decline: {decline:.4f} | {key_preview}")
    
    # Additional insights
    print("\n🔍 ADDITIONAL INSIGHTS:")
    print("-" * 40)
    
    # Analyze learning progression
    if analysis['global_stats'].get('rewards_by_step'):
        steps = sorted([int(s) for s in analysis['global_stats']['rewards_by_step'].keys()])
        if len(steps) > 10:
            early_avg = np.mean([np.mean(analysis['global_stats']['rewards_by_step'][str(s)]) 
                                for s in steps[:10]])
            late_avg = np.mean([np.mean(analysis['global_stats']['rewards_by_step'][str(s)]) 
                               for s in steps[-10:]])
            improvement = late_avg - early_avg
            pct_improvement = (improvement / early_avg * 100) if early_avg > 0 else 0
            
            print(f"Average reward improvement: {improvement:.3f} ({pct_improvement:+.1f}%)")
            print(f"  Early training avg: {early_avg:.3f}")
            print(f"  Late training avg: {late_avg:.3f}")
    
    # Problem diversity
    print(f"\nProblem diversity metrics:")
    mean_rewards = [stats['mean_reward'] for stats in analysis['problem_stats'].values()]
    print(f"  Reward range: [{min(mean_rewards):.3f}, {max(mean_rewards):.3f}]")
    print(f"  Reward std dev: {np.std(mean_rewards):.3f}")
    print(f"  Gini coefficient: {calculate_gini(mean_rewards):.3f}")
    
    # Save detailed prompt analysis
    output_filename = f'individual_problem_analysis_{run_id}_results.json'
    save_analysis_results(analysis, output_filename)
    
    # Generate detailed prompt analysis by category
    prompt_analysis_filename = f'prompt_analysis_{run_id}.txt'
    generate_prompt_analysis(learning_progression_insights, analysis['problem_stats'], prompt_analysis_filename)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()