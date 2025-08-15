"""Analyze terminal_exploration_depth_reward performance in WandB run."""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from wandb_utils import get_run, get_table_data

# Initialize wandb - try different project names
entity = "bjornlau"
project = "CodeRepairRL"
run_id = "z5iaa297"

try:
    run = get_run(entity, project, run_id)
except Exception as e:
    print(f"Failed with {entity}/{project}, error: {e}")
    print("Run not found. Please check if the run ID is correct.")
    sys.exit(1)

# Set up output directory
output_dir = Path(__file__).parent.parent.parent / "figures" / "plots" / "analysis"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Analyzing run: {run.name} ({run.id})")
print(f"State: {run.state}")
print()

# Get all tables from the run
tables = run.logged_artifacts()
reward_data = []

print("Searching for terminal reward data in tables...")
for artifact in tables:
    if artifact.type == 'run_table':
        try:
            table = artifact.get("table")
            if table and hasattr(table, 'get_dataframe'):
                df = table.get_dataframe()
                
                # Look for terminal exploration depth reward columns
                depth_cols = [col for col in df.columns if 'terminal_exploration_depth_reward' in col.lower()]
                
                if depth_cols:
                    print(f"\nFound reward data in {artifact.name}:")
                    print(f"  Columns: {depth_cols}")
                    print(f"  Shape: {df.shape}")
                    
                    for col in depth_cols:
                        if col in df.columns:
                            rewards = df[col].dropna()
                            if len(rewards) > 0:
                                reward_data.extend(rewards.tolist())
                                
                                # Analyze this specific table
                                print(f"\n  Stats for {col}:")
                                print(f"    Count: {len(rewards)}")
                                print(f"    Mean: {rewards.mean():.4f}")
                                print(f"    Std: {rewards.std():.4f}")
                                print(f"    Min: {rewards.min():.4f}")
                                print(f"    Max: {rewards.max():.4f}")
                                print(f"    Zeros: {(rewards == 0).sum()} ({(rewards == 0).mean()*100:.1f}%)")
                                
                                # Show value distribution
                                value_counts = pd.cut(rewards, bins=[0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).value_counts().sort_index()
                                print(f"    Distribution:")
                                for interval, count in value_counts.items():
                                    print(f"      {interval}: {count} ({count/len(rewards)*100:.1f}%)")
                                
                                # Look for specific patterns in high rewards
                                high_rewards = rewards[rewards > 0.4]
                                if len(high_rewards) > 0:
                                    print(f"\n  High rewards (>0.4): {len(high_rewards)} instances")
                                    # Try to get corresponding completion data if available
                                    if 'completion' in df.columns:
                                        high_reward_indices = rewards[rewards > 0.4].index
                                        for idx in high_reward_indices[:3]:  # Show first 3 examples
                                            if idx in df.index:
                                                print(f"\n    Example {idx}: Reward = {rewards[idx]:.3f}")
                                                completion = df.loc[idx, 'completion'] if 'completion' in df.columns else "N/A"
                                                if isinstance(completion, str) and len(completion) > 200:
                                                    # Look for terminal commands
                                                    import re
                                                    tool_calls = re.findall(r'<tool_call>.*?</tool_call>', completion, re.DOTALL)
                                                    if tool_calls:
                                                        print(f"      Found {len(tool_calls)} tool calls")
                                                        # Extract commands
                                                        for tc in tool_calls[:5]:
                                                            cmd_match = re.search(r'"cmd":\s*"([^"]+)"', tc)
                                                            if cmd_match:
                                                                print(f"        - {cmd_match.group(1)[:80]}")
                    
        except Exception as e:
            print(f"Error processing {artifact.name}: {e}")
            continue

if reward_data:
    rewards_array = np.array(reward_data)
    
    print(f"\n{'='*60}")
    print(f"Overall Terminal Exploration Depth Reward Analysis")
    print(f"{'='*60}")
    print(f"Total samples: {len(rewards_array)}")
    print(f"Mean: {rewards_array.mean():.4f}")
    print(f"Median: {np.median(rewards_array):.4f}")
    print(f"Std: {rewards_array.std():.4f}")
    print(f"Min: {rewards_array.min():.4f}")
    print(f"Max: {rewards_array.max():.4f}")
    print(f"Zeros: {(rewards_array == 0).sum()} ({(rewards_array == 0).mean()*100:.1f}%)")
    
    # Analyze reward components (based on the function logic)
    # Possible values based on bonuses:
    # 0.25 (chain), 0.20 (success/precise), 0.15 (scoped)
    reward_levels = defaultdict(int)
    for r in rewards_array:
        if r == 0:
            reward_levels['zero'] += 1
        elif 0 < r <= 0.15:
            reward_levels['low (≤0.15)'] += 1
        elif 0.15 < r <= 0.25:
            reward_levels['medium (0.15-0.25)'] += 1
        elif 0.25 < r <= 0.40:
            reward_levels['good (0.25-0.40)'] += 1
        elif 0.40 < r <= 0.60:
            reward_levels['high (0.40-0.60)'] += 1
        else:
            reward_levels['very high (>0.60)'] += 1
    
    print("\nReward Level Distribution:")
    for level, count in sorted(reward_levels.items()):
        print(f"  {level}: {count} ({count/len(rewards_array)*100:.1f}%)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(rewards_array, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(rewards_array.mean(), color='red', linestyle='--', label=f'Mean: {rewards_array.mean():.3f}')
    ax.axvline(np.median(rewards_array), color='green', linestyle='--', label=f'Median: {np.median(rewards_array):.3f}')
    ax.set_xlabel('Reward Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Terminal Exploration Depth Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. CDF
    ax = axes[0, 1]
    sorted_rewards = np.sort(rewards_array)
    cdf = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
    ax.plot(sorted_rewards, cdf, linewidth=2)
    ax.set_xlabel('Reward Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.grid(True, alpha=0.3)
    
    # Mark key percentiles
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(rewards_array, p)
        ax.axvline(val, color='gray', linestyle=':', alpha=0.5)
        ax.text(val, 0.05, f'{p}%: {val:.3f}', rotation=90, fontsize=8)
    
    # 3. Box plot with detailed breakdown
    ax = axes[1, 0]
    # Group rewards into buckets for analysis
    buckets = []
    labels = []
    
    if (rewards_array == 0).any():
        buckets.append(rewards_array[rewards_array == 0])
        labels.append('Zero\n(No bonus)')
    
    ranges = [(0.001, 0.15), (0.15, 0.25), (0.25, 0.40), (0.40, 1.0)]
    range_labels = ['Low\n(Single small)', 'Medium\n(Single bonus)', 'Good\n(Multi bonus)', 'High\n(Chain/Multi)']
    
    for (low, high), label in zip(ranges, range_labels):
        mask = (rewards_array >= low) & (rewards_array < high)
        if mask.any():
            buckets.append(rewards_array[mask])
            labels.append(label)
    
    if buckets:
        bp = ax.boxplot(buckets, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_ylabel('Reward Value')
        ax.set_title('Reward Distribution by Level')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Time series if we can extract temporal data
    ax = axes[1, 1]
    # Simple running average
    window = min(100, len(rewards_array) // 10)
    if window > 1:
        running_mean = pd.Series(rewards_array).rolling(window=window, min_periods=1).mean()
        ax.plot(running_mean, alpha=0.8, label=f'Running Mean (window={window})')
        ax.fill_between(range(len(running_mean)), 0, running_mean, alpha=0.3)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Reward Value')
        ax.set_title('Reward Trend Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Terminal Exploration Depth Reward Analysis\nRun: {run.id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f"deep_reward_analysis_{run.id}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    
    # Analysis conclusions
    print(f"\n{'='*60}")
    print("ANALYSIS CONCLUSIONS:")
    print(f"{'='*60}")
    
    zero_pct = (rewards_array == 0).mean() * 100
    if zero_pct > 50:
        print(f"⚠️  HIGH ZERO RATE ({zero_pct:.1f}%): The reward may be too strict or agents aren't exhibiting the target behaviors")
    
    if rewards_array.max() < 0.5:
        print(f"⚠️  LOW MAX REWARD ({rewards_array.max():.3f}): Full bonus combinations are rare or never achieved")
    
    if rewards_array.std() < 0.1:
        print(f"⚠️  LOW VARIANCE ({rewards_array.std():.3f}): Rewards lack diversity, may not provide good learning signal")
    
    # Check if specific bonuses are being triggered
    typical_bonuses = [0.15, 0.20, 0.25, 0.35, 0.40, 0.45]
    bonus_hits = {}
    for bonus in typical_bonuses:
        hits = np.sum(np.abs(rewards_array - bonus) < 0.02)  # Within 0.02 of target
        bonus_hits[bonus] = hits
    
    print("\nLikely Bonus Patterns Detected:")
    for bonus, hits in sorted(bonus_hits.items()):
        if hits > len(rewards_array) * 0.01:  # At least 1% of samples
            print(f"  ~{bonus:.2f}: {hits} times ({hits/len(rewards_array)*100:.1f}%) - ", end="")
            if bonus == 0.15:
                print("Scoped search")
            elif bonus == 0.20:
                print("Success ratio OR Precise reads")
            elif bonus == 0.25:
                print("Chain bonus")
            elif bonus == 0.35:
                print("Likely Scoped + Success/Precise")
            elif bonus == 0.40:
                print("Likely Success + Precise")
            elif bonus == 0.45:
                print("Likely Chain + Success/Precise")
            else:
                print("Multiple bonuses")
    
    # Performance assessment
    mean_reward = rewards_array.mean()
    if mean_reward < 0.1:
        print(f"\n❌ POOR PERFORMANCE: Mean reward {mean_reward:.3f} suggests reward is not effectively shaping behavior")
    elif mean_reward < 0.2:
        print(f"\n⚠️  WEAK PERFORMANCE: Mean reward {mean_reward:.3f} indicates limited learning signal")
    else:
        print(f"\n✅ REASONABLE PERFORMANCE: Mean reward {mean_reward:.3f} provides meaningful signal")

else:
    print("\n❌ No terminal exploration depth reward data found in run!")