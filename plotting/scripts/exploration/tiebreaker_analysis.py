#!/usr/bin/env python3
"""
Re-analyze the deep reward as a GRPO tiebreaker rather than primary signal.
Focus on: discrimination power, consistency, and avoiding reward collapse.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import re
import json
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from wandb_utils import get_run
from utils.table_parser import TableExtractor
from collections import defaultdict, Counter

# Configuration
ENTITY = "assert-kth"
PROJECT = "SWE-Gym-GRPO"
RUN_ID = "z5iaa297"

def get_total_reward(row) -> float:
    """Extract the total reward (main + deep reward components)."""
    try:
        reward_str = row.get('Reward', '0')
        return float(reward_str) if reward_str else 0.0
    except:
        return 0.0

def get_deep_reward(row) -> float:
    """Extract just the deep reward component."""
    deep_reward_cols = [col for col in row.index if 'terminal_exploration_depth' in col.lower()]
    if deep_reward_cols:
        try:
            reward_str = row.get(deep_reward_cols[0], '0')
            return float(reward_str) if reward_str else 0.0
        except:
            return 0.0
    return 0.0

def analyze_reward_discrimination(samples):
    """Analyze how well the deep reward discriminates between similar cases."""
    
    # Group by total reward to see tiebreaker effectiveness
    reward_groups = defaultdict(list)
    for sample in samples:
        # Round total reward to see groups
        rounded_total = round(sample['total_reward'], 2)
        reward_groups[rounded_total].append(sample)
    
    print(f"Reward Discrimination Analysis:")
    print(f"  Total reward groups: {len(reward_groups)}")
    
    # Focus on groups with multiple samples (where tiebreaking matters)
    tied_groups = {r: samples for r, samples in reward_groups.items() if len(samples) > 1}
    
    print(f"  Groups with ties: {len(tied_groups)}")
    
    discrimination_analysis = {
        'total_tied_cases': sum(len(samples) for samples in tied_groups.values()),
        'well_discriminated': 0,
        'poorly_discriminated': 0,
        'examples': []
    }
    
    for total_reward, group_samples in tied_groups.items():
        if len(group_samples) < 2:
            continue
            
        deep_rewards = [s['deep_reward'] for s in group_samples]
        deep_reward_variance = np.var(deep_rewards)
        
        print(f"\n  Group with total reward {total_reward} ({len(group_samples)} samples):")
        print(f"    Deep rewards: {[f'{r:.3f}' for r in deep_rewards]}")
        print(f"    Deep reward variance: {deep_reward_variance:.4f}")
        
        if deep_reward_variance > 0.01:  # Meaningful discrimination
            discrimination_analysis['well_discriminated'] += len(group_samples)
            status = "✅ Good discrimination"
        else:
            discrimination_analysis['poorly_discriminated'] += len(group_samples)
            status = "❌ Poor discrimination"
        
        print(f"    {status}")
        
        # Store example for detailed analysis
        if len(discrimination_analysis['examples']) < 3:
            discrimination_analysis['examples'].append({
                'total_reward': total_reward,
                'samples': group_samples[:3],  # First 3 samples
                'variance': deep_reward_variance
            })
    
    return discrimination_analysis

def check_reward_collapse(samples):
    """Check if deep rewards are collapsing to similar values (bad for GRPO)."""
    deep_rewards = [s['deep_reward'] for s in samples]
    
    # Count frequency of each reward value
    reward_counts = Counter([round(r, 3) for r in deep_rewards])
    
    print(f"\nReward Collapse Analysis:")
    print(f"  Unique deep reward values: {len(reward_counts)}")
    print(f"  Total samples: {len(deep_rewards)}")
    print(f"  Entropy: {calculate_entropy(list(reward_counts.values())):.3f}")
    
    # Check for over-concentration
    most_common = reward_counts.most_common(5)
    print(f"  Most common values:")
    for value, count in most_common:
        percentage = count / len(deep_rewards) * 100
        print(f"    {value}: {count} ({percentage:.1f}%)")
    
    # Check if too concentrated
    top_value_pct = most_common[0][1] / len(deep_rewards) * 100
    if top_value_pct > 50:
        print(f"  ❌ REWARD COLLAPSE: {top_value_pct:.1f}% of samples have same reward")
        return False
    elif top_value_pct > 30:
        print(f"  ⚠️  CONCENTRATION WARNING: {top_value_pct:.1f}% have same reward")
        return True
    else:
        print(f"  ✅ Good distribution: top value is {top_value_pct:.1f}%")
        return True

def calculate_entropy(counts):
    """Calculate entropy of reward distribution."""
    total = sum(counts)
    if total == 0:
        return 0
    
    entropy = 0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return entropy

def analyze_grpo_signal_quality(samples):
    """Analyze quality of signal for GRPO hill-climbing."""
    deep_rewards = [s['deep_reward'] for s in samples]
    
    print(f"\nGRPO Signal Quality Analysis:")
    
    # Basic statistics
    mean_reward = np.mean(deep_rewards)
    std_reward = np.std(deep_rewards)
    min_reward = np.min(deep_rewards)
    max_reward = np.max(deep_rewards)
    
    print(f"  Mean: {mean_reward:.3f}")
    print(f"  Std: {std_reward:.3f}")
    print(f"  Range: [{min_reward:.3f}, {max_reward:.3f}]")
    print(f"  Coefficient of variation: {std_reward/mean_reward:.3f}")
    
    # Check for good gradient
    signal_to_noise = (max_reward - min_reward) / std_reward if std_reward > 0 else 0
    print(f"  Signal-to-noise ratio: {signal_to_noise:.3f}")
    
    # Check percentile spread (important for GRPO)
    percentiles = [10, 25, 50, 75, 90]
    pct_values = [np.percentile(deep_rewards, p) for p in percentiles]
    print(f"  Percentiles: {dict(zip(percentiles, [f'{v:.3f}' for v in pct_values]))}")
    
    # Gradual progression is good for hill-climbing
    pct_gaps = [pct_values[i+1] - pct_values[i] for i in range(len(pct_values)-1)]
    avg_gap = np.mean(pct_gaps)
    print(f"  Average percentile gap: {avg_gap:.3f}")
    
    # Assessment
    quality_score = 0
    assessments = []
    
    if std_reward > 0.05:
        quality_score += 1
        assessments.append("✅ Good variance for discrimination")
    else:
        assessments.append("❌ Low variance - poor discrimination")
    
    if signal_to_noise > 2:
        quality_score += 1  
        assessments.append("✅ Good signal-to-noise ratio")
    else:
        assessments.append("❌ Poor signal-to-noise ratio")
    
    if avg_gap > 0.02:
        quality_score += 1
        assessments.append("✅ Good gradual progression")
    else:
        assessments.append("❌ Compressed percentiles")
    
    if mean_reward > 0.1:
        quality_score += 1
        assessments.append("✅ Reasonable baseline reward")
    else:
        assessments.append("❌ Very low baseline rewards")
    
    print(f"\n  GRPO Signal Quality: {quality_score}/4")
    for assessment in assessments:
        print(f"    {assessment}")
    
    return quality_score

def main():
    """Main analysis function."""
    print("="*80)
    print("Deep Reward Analysis: GRPO Tiebreaker Effectiveness")
    print("="*80)
    
    # Get run
    try:
        run = get_run(ENTITY, PROJECT, RUN_ID)
    except Exception as e:
        print(f"❌ Failed to load run: {e}")
        return
    
    # Extract tables
    print("\nExtracting tables from run...")
    extractor = TableExtractor(max_workers=5)
    tables = extractor.extract_all_training_tables(run, max_tables=25)
    
    if not tables:
        print("❌ No tables found")
        return
    
    print(f"✅ Extracted {len(tables)} tables")
    
    # Collect all samples with both total and deep rewards
    samples = []
    
    for table_idx, df in enumerate(tables):
        deep_reward_cols = [col for col in df.columns if 'terminal_exploration_depth' in col.lower()]
        
        if not deep_reward_cols:
            continue
        
        for rollout_idx in range(len(df)):
            try:
                row = df.iloc[rollout_idx]
                
                total_reward = get_total_reward(row)
                deep_reward = get_deep_reward(row)
                
                samples.append({
                    'table': table_idx,
                    'rollout': rollout_idx,
                    'total_reward': total_reward,
                    'deep_reward': deep_reward,
                    'completion': row.get('Completion', '')[:200]  # Preview
                })
                
            except Exception as e:
                continue
    
    if not samples:
        print("❌ No valid samples found")
        return
    
    print(f"Collected {len(samples)} samples for analysis")
    
    # 1. Analyze discrimination power
    print("\n" + "="*80)
    print("1. DISCRIMINATION POWER ANALYSIS")
    print("="*80)
    
    discrimination = analyze_reward_discrimination(samples)
    
    # 2. Check for reward collapse
    print("\n" + "="*80) 
    print("2. REWARD COLLAPSE CHECK")
    print("="*80)
    
    no_collapse = check_reward_collapse(samples)
    
    # 3. Analyze GRPO signal quality
    print("\n" + "="*80)
    print("3. GRPO HILL-CLIMBING SIGNAL QUALITY")
    print("="*80)
    
    signal_quality = analyze_grpo_signal_quality(samples)
    
    # 4. Show detailed examples of tiebreaking
    print("\n" + "="*80)
    print("4. DETAILED TIEBREAKING EXAMPLES")
    print("="*80)
    
    for i, example in enumerate(discrimination['examples'][:2], 1):
        print(f"\nExample {i}: Total Reward = {example['total_reward']}")
        print(f"Deep reward variance: {example['variance']:.4f}")
        
        for j, sample in enumerate(example['samples'], 1):
            print(f"  Sample {j}:")
            print(f"    Deep reward: {sample['deep_reward']:.3f}")
            print(f"    Completion preview: {sample['completion']}")
    
    # 5. Final assessment as tiebreaker
    print("\n" + "="*80)
    print("5. TIEBREAKER EFFECTIVENESS ASSESSMENT")
    print("="*80)
    
    effectiveness_score = 0
    
    # Criterion 1: Discrimination power
    discrimination_ratio = discrimination['well_discriminated'] / max(1, discrimination['total_tied_cases'])
    if discrimination_ratio > 0.7:
        effectiveness_score += 1
        print("✅ GOOD: Discriminates well in tied cases")
    else:
        print(f"❌ POOR: Only discriminates {discrimination_ratio:.1%} of tied cases")
    
    # Criterion 2: No reward collapse
    if no_collapse:
        effectiveness_score += 1
        print("✅ GOOD: No reward collapse detected")
    else:
        print("❌ POOR: Reward collapse detected")
    
    # Criterion 3: Signal quality
    if signal_quality >= 3:
        effectiveness_score += 1
        print("✅ GOOD: Strong GRPO signal quality")
    elif signal_quality >= 2:
        print("⚠️  OKAY: Moderate GRPO signal quality")
    else:
        print("❌ POOR: Weak GRPO signal quality")
    
    # Criterion 4: Meaningful baseline
    mean_deep = np.mean([s['deep_reward'] for s in samples])
    if mean_deep > 0.15:
        effectiveness_score += 1
        print("✅ GOOD: Meaningful baseline prevents flat rewards")
    else:
        print("❌ POOR: Low baseline, risk of all-zero scenarios")
    
    print(f"\nOVERALL TIEBREAKER EFFECTIVENESS: {effectiveness_score}/4")
    
    if effectiveness_score >= 3:
        print("🎯 EXCELLENT: Deep reward serves as effective GRPO tiebreaker")
    elif effectiveness_score >= 2:
        print("⚠️  ADEQUATE: Deep reward works but has room for improvement")
    else:
        print("❌ INADEQUATE: Deep reward fails as effective tiebreaker")
    
    print("\n✅ Tiebreaker analysis complete!")

if __name__ == "__main__":
    main()