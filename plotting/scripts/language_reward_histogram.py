#!/usr/bin/env python3
"""
Plot average reward in early vs late training stages as a grouped histogram per language.
Similar to Sankey diagram approach: first 20% of training steps = early, last 20% = late.
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from wandb_utils import get_run
from plot_config import ENTITY, PROJECT, RUN_ID, get_output_filename, setup_plotting_style
from utils.table_parser import TableExtractor

# Language file extension patterns
LANGUAGE_PATTERNS = {
    'Python': [r'\.py$'],
    'Rust': [r'\.rs$'],
    'Java': [r'\.java$'],
    'PHP': [r'\.php$'],
    'Ruby': [r'\.rb$'],
    'JavaScript': [r'\.js$', r'\.jsx$', r'\.mjs$'],
    'TypeScript': [r'\.ts$', r'\.tsx$'],
    'Go': [r'\.go$'],
    'C': [r'\.c$', r'\.h$'],
    'C++': [r'\.cpp$', r'\.cc$', r'\.cxx$', r'\.hpp$', r'\.hh$', r'\.hxx$'],
}

def detect_language_from_patch(patch: str) -> str:
    """
    Detect programming language from git diff patch by examining file extensions.

    Args:
        patch: Git diff patch string

    Returns:
        Detected language name, or None if no match
    """
    # Extract filenames from diff headers
    file_patterns = [
        r'diff --git a/[^\s]+ b/([^\s]+)',
        r'\+\+\+ b/([^\s]+)',
        r'--- a/([^\s]+)'
    ]

    filenames = []
    for pattern in file_patterns:
        matches = re.findall(pattern, patch)
        filenames.extend(matches)

    # Try to match file extensions to languages
    for filename in filenames:
        for lang, patterns in LANGUAGE_PATTERNS.items():
            for ext_pattern in patterns:
                if re.search(ext_pattern, filename, re.IGNORECASE):
                    return lang

    return None

def main():
    """Create early vs late training reward histogram by language."""
    parser = argparse.ArgumentParser(description='Plot early vs late training rewards by language')
    parser.add_argument('--run-id', type=str, default=RUN_ID,
                        help=f'WandB run ID (default: {RUN_ID})')
    parser.add_argument('--merge-with', type=str, default=None,
                        help='Optional second run ID to merge with')
    parser.add_argument('--early-pct', type=float, default=0.2,
                        help='Percentage of steps to consider "early" (default: 0.2 = 20%%)')
    parser.add_argument('--late-pct', type=float, default=0.2,
                        help='Percentage of steps to consider "late" (default: 0.2 = 20%%)')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='Minimum samples per language to include (default: 5)')
    args = parser.parse_args()

    print("="*60)
    print("Language Reward Histogram: Early vs Late Training")
    print("="*60)

    # Get run and extract tables
    print(f"\nLoading run {args.run_id}...")

    if args.merge_with:
        print(f"Merging with {args.merge_with}...")
        # Get both runs and extract tables
        run1 = get_run(ENTITY, PROJECT, args.run_id)
        run2 = get_run(ENTITY, PROJECT, args.merge_with)

        extractor = TableExtractor()
        tables1 = extractor.extract_all_training_tables(run1)
        tables2 = extractor.extract_all_training_tables(run2)

        # Determine which is the continuation (higher max step)
        max_step1 = max(
            row.get('global_step', 0)
            for table in tables1
            for _, row in table.iterrows()
        ) if tables1 else 0

        max_step2 = max(
            row.get('global_step', 0)
            for table in tables2
            for _, row in table.iterrows()
        ) if tables2 else 0

        # The run with higher max step is the continuation
        if max_step1 > max_step2:
            # run1 is continuation, run2 is earlier
            print(f"  {args.run_id} is continuation (max step: {max_step1})")
            print(f"  {args.merge_with} is earlier run (max step: {max_step2})")
            # Offset continuation (run1) by earlier run's max step
            for table in tables1:
                if 'global_step' in table.columns:
                    table['global_step'] = table['global_step'] + max_step2 + 1
            tables = tables2 + tables1  # Earlier first, then continuation
            run = run1
        else:
            # run2 is continuation, run1 is earlier
            print(f"  {args.merge_with} is continuation (max step: {max_step2})")
            print(f"  {args.run_id} is earlier run (max step: {max_step1})")
            # Offset continuation (run2) by earlier run's max step
            for table in tables2:
                if 'global_step' in table.columns:
                    table['global_step'] = table['global_step'] + max_step1 + 1
            tables = tables1 + tables2  # Earlier first, then continuation
            run = run2
    else:
        run = get_run(ENTITY, PROJECT, args.run_id)
        extractor = TableExtractor()
        tables = extractor.extract_all_training_tables(run)

    if not tables:
        print("❌ No tables found")
        return

    # Extract language-specific rewards with global steps
    print("\nExtracting language-specific rewards...")
    raw_data = []

    for table in tables:
        for idx, row in table.iterrows():
            try:
                patch = row.get('Patch', '')
                reward_str = row.get('Unified_diff_similarity_reward_func', '0.0')
                global_step = row.get('global_step', 0)

                # Parse reward
                try:
                    reward = float(reward_str)
                except (ValueError, TypeError):
                    reward = 0.0

                # Detect language
                language = detect_language_from_patch(patch)
                if language is None:
                    continue

                # Store data point
                raw_data.append({
                    'step': global_step,
                    'language': language,
                    'reward': reward
                })
            except Exception as e:
                continue

    if not raw_data:
        print("❌ No data extracted")
        return

    df = pd.DataFrame(raw_data)
    print(f"Extracted {len(df)} rollouts across {df['language'].nunique()} languages")

    # Determine early and late step ranges
    min_step = df['step'].min()
    max_step = df['step'].max()
    step_range = max_step - min_step

    early_end = min_step + (step_range * args.early_pct)
    late_start = max_step - (step_range * args.late_pct)

    print(f"\nStep ranges:")
    print(f"  Total: {min_step} to {max_step} ({step_range} steps)")
    print(f"  Early: {min_step} to {early_end:.0f} (first {args.early_pct*100:.0f}%)")
    print(f"  Late: {late_start:.0f} to {max_step} (last {args.late_pct*100:.0f}%)")

    # Split data into early and late
    df_early = df[df['step'] <= early_end]
    df_late = df[df['step'] >= late_start]

    print(f"\nRollout counts:")
    print(f"  Early: {len(df_early)} rollouts")
    print(f"  Late: {len(df_late)} rollouts")

    # Calculate average reward per language for early and late
    early_rewards = df_early.groupby('language')['reward'].agg(['mean', 'count']).reset_index()
    late_rewards = df_late.groupby('language')['reward'].agg(['mean', 'count']).reset_index()

    early_rewards.columns = ['language', 'early_reward', 'early_count']
    late_rewards.columns = ['language', 'late_reward', 'late_count']

    # Merge and filter by minimum samples
    merged = pd.merge(early_rewards, late_rewards, on='language', how='outer').fillna(0)
    merged = merged[
        (merged['early_count'] >= args.min_samples) &
        (merged['late_count'] >= args.min_samples)
    ]

    if len(merged) == 0:
        print(f"❌ No languages with at least {args.min_samples} samples in both early and late stages")
        return

    # Sort by late reward for better visualization
    merged = merged.sort_values('late_reward', ascending=False)

    print(f"\nLanguages with sufficient data ({args.min_samples}+ samples each stage):")
    print(f"  {'Language':<12} {'Early Reward':>12} {'Early Count':>12} {'Late Reward':>12} {'Late Count':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for _, row in merged.iterrows():
        print(f"  {row['language']:12} {row['early_reward']:12.3f} {row['early_count']:12.0f} {row['late_reward']:12.3f} {row['late_count']:12.0f}")

    # Create grouped bar chart
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    languages = merged['language'].values
    early = merged['early_reward'].values
    late = merged['late_reward'].values

    x = np.arange(len(languages))
    width = 0.35

    bars1 = ax.bar(x - width/2, early, width, label='Early Training', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x + width/2, late, width, label='Late Training', color='#E74C3C', alpha=0.8)

    # Styling
    ax.set_xlabel('Programming Language', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=14, fontweight='bold')
    ax.set_title(f'{run.name}: Reward by Language (Early vs Late Training)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(languages, rotation=45, ha='right')
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()

    # Save
    output_path = get_output_filename(
        f"language_reward_histogram_early{int(args.early_pct*100)}_late{int(args.late_pct*100)}",
        args.run_id,
        plot_type="analysis"
    )
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to: {output_path}.png")

    print(f"\n✅ Language reward histogram created successfully!")

if __name__ == "__main__":
    main()
