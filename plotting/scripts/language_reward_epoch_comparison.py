#!/usr/bin/env python3
"""
Compare reward improvement for the same prompts across epochs.
Tracks the first 20 unique prompts from epoch 1 and compares their rewards in epoch 2.
Each table contains 8 rollouts for the same prompt.
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
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
    """Detect programming language from git diff patch."""
    file_patterns = [
        r'diff --git a/[^\s]+ b/([^\s]+)',
        r'\+\+\+ b/([^\s]+)',
        r'--- a/([^\s]+)'
    ]

    filenames = []
    for pattern in file_patterns:
        matches = re.findall(pattern, patch)
        filenames.extend(matches)

    for filename in filenames:
        for lang, patterns in LANGUAGE_PATTERNS.items():
            for ext_pattern in patterns:
                if re.search(ext_pattern, filename, re.IGNORECASE):
                    return lang
    return None

def main():
    """Compare same prompts across epochs."""
    parser = argparse.ArgumentParser(description='Compare rewards for same prompts across epochs')
    parser.add_argument('--run-id', type=str, default=RUN_ID,
                        help=f'WandB run ID (default: {RUN_ID})')
    parser.add_argument('--merge-with', type=str, default=None,
                        help='Optional second run ID to merge with')
    parser.add_argument('--num-prompts', type=int, default=20,
                        help='Number of unique prompts to track (default: 20)')
    parser.add_argument('--disjoint', action='store_true',
                        help='Use disjoint sets: 50%% epoch 1, 50%% epoch 2 (avoids overfitting concerns)')
    args = parser.parse_args()

    print("="*60)
    print("Language Reward: Same Prompts Across Epochs")
    print("="*60)

    # Get run and extract tables
    print(f"\nLoading run {args.run_id}...")

    if args.merge_with:
        print(f"Merging with {args.merge_with}...")
        run1 = get_run(ENTITY, PROJECT, args.run_id)
        run2 = get_run(ENTITY, PROJECT, args.merge_with)

        extractor = TableExtractor()
        tables1 = extractor.extract_all_training_tables(run1)
        tables2 = extractor.extract_all_training_tables(run2)

        # Determine which is continuation
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

        if max_step1 > max_step2:
            print(f"  {args.run_id} is continuation (max step: {max_step1})")
            print(f"  {args.merge_with} is earlier run (max step: {max_step2})")
            for table in tables1:
                if 'global_step' in table.columns:
                    table['global_step'] = table['global_step'] + max_step2 + 1
            tables = tables2 + tables1
            run = run1
        else:
            print(f"  {args.merge_with} is continuation (max step: {max_step2})")
            print(f"  {args.run_id} is earlier run (max step: {max_step1})")
            for table in tables2:
                if 'global_step' in table.columns:
                    table['global_step'] = table['global_step'] + max_step1 + 1
            tables = tables1 + tables2
            run = run2
    else:
        run = get_run(ENTITY, PROJECT, args.run_id)
        extractor = TableExtractor()
        tables = extractor.extract_all_training_tables(run)

    if not tables:
        print("❌ No tables found")
        return

    # Track prompts and their rewards
    print("\nTracking prompts across epochs...")
    prompt_data = defaultdict(lambda: {'epoch1': [], 'epoch2': [], 'language': None, 'patch': None})
    seen_prompts = set()
    first_n_prompts = []

    for table in tables:
        for idx, row in table.iterrows():
            try:
                prompt = row.get('Prompt', '')
                if not prompt:
                    continue

                patch = row.get('Patch', '')
                reward_str = row.get('Unified_diff_similarity_reward_func', '0.0')
                global_step = row.get('global_step', 0)

                # Parse reward
                try:
                    reward = float(reward_str)
                except (ValueError, TypeError):
                    reward = 0.0

                # Track first N unique prompts
                if prompt not in seen_prompts:
                    seen_prompts.add(prompt)
                    if len(first_n_prompts) < args.num_prompts:
                        first_n_prompts.append(prompt)
                        # Detect language for this prompt
                        language = detect_language_from_patch(patch)
                        prompt_data[prompt]['language'] = language
                        prompt_data[prompt]['patch'] = patch

                # Only track data for the first N prompts
                if prompt in first_n_prompts:
                    # Heuristic: first ~280 steps is epoch 1, rest is epoch 2
                    # Since we have ~553 steps total and ~2 epochs
                    if global_step < 280:
                        prompt_data[prompt]['epoch1'].append(reward)
                    else:
                        prompt_data[prompt]['epoch2'].append(reward)

            except Exception as e:
                continue

    print(f"Tracked {len(first_n_prompts)} unique prompts")

    # Count prompts by language
    language_prompt_counts = defaultdict(int)
    for prompt in first_n_prompts:
        lang = prompt_data[prompt]['language']
        if lang:
            language_prompt_counts[lang] += 1

    print(f"\nPrompt distribution:")
    for lang, count in sorted(language_prompt_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count} prompts")

    # Check epoch split for a few prompts
    print(f"\nEpoch split validation (first 5 prompts):")
    for i, prompt in enumerate(first_n_prompts[:5]):
        data = prompt_data[prompt]
        print(f"  Prompt {i+1} ({data['language']}): Epoch 1 has {len(data['epoch1'])} rollouts, Epoch 2 has {len(data['epoch2'])} rollouts")

    # Calculate average rewards per language for epoch 1 vs epoch 2
    language_rewards = defaultdict(lambda: {'epoch1': [], 'epoch2': []})

    if args.disjoint:
        # Disjoint mode: split prompts in temporal order (first 50% vs last 50%)
        print(f"\n🔀 Disjoint mode: Comparing early training (first 50%) vs late training (last 50%)")

        # Split prompts by language, preserving temporal order
        prompts_by_language = defaultdict(list)
        for prompt in first_n_prompts:
            data = prompt_data[prompt]
            language = data['language']
            if language and len(data['epoch1']) > 0 and len(data['epoch2']) > 0:
                prompts_by_language[language].append(prompt)

        # For each language, split in order (no shuffling)
        for lang, prompts in prompts_by_language.items():
            # Split prompts in temporal order
            split_idx = len(prompts) // 2

            epoch1_prompts = prompts[:split_idx]  # First 50% appear earlier in training
            epoch2_prompts = prompts[split_idx:]  # Last 50% appear later in training

            print(f"  {lang}: {len(epoch1_prompts)} early prompts (epoch 1), {len(epoch2_prompts)} late prompts (epoch 2)")

            # Use epoch 1 data for first half (early training)
            for prompt in epoch1_prompts:
                data = prompt_data[prompt]
                avg = np.mean(data['epoch1'])
                language_rewards[lang]['epoch1'].append(avg)

            # Use epoch 2 data for second half (late training)
            for prompt in epoch2_prompts:
                data = prompt_data[prompt]
                avg = np.mean(data['epoch2'])
                language_rewards[lang]['epoch2'].append(avg)
    else:
        # Original mode: same prompts across epochs
        for prompt in first_n_prompts:
            data = prompt_data[prompt]
            language = data['language']

            if language is None:
                continue

            epoch1_rewards = data['epoch1']
            epoch2_rewards = data['epoch2']

            if len(epoch1_rewards) > 0 and len(epoch2_rewards) > 0:
                # Average across the 8 rollouts per epoch
                avg_epoch1 = np.mean(epoch1_rewards)
                avg_epoch2 = np.mean(epoch2_rewards)

                language_rewards[language]['epoch1'].append(avg_epoch1)
                language_rewards[language]['epoch2'].append(avg_epoch2)

    # Calculate mean reward per language
    results = []
    for lang, data in language_rewards.items():
        if len(data['epoch1']) >= 2:  # At least 2 prompts per language
            mean_epoch1 = np.mean(data['epoch1'])
            mean_epoch2 = np.mean(data['epoch2'])
            count = len(data['epoch1'])
            results.append({
                'language': lang,
                'epoch1': mean_epoch1,
                'epoch2': mean_epoch2,
                'count': count,
                'improvement': mean_epoch2 - mean_epoch1
            })

    if not results:
        print("❌ No languages with sufficient data")
        return

    # Convert to DataFrame and sort by epoch2 reward
    df = pd.DataFrame(results).sort_values('epoch2', ascending=False)

    print(f"\nLanguage performance across epochs (tracking first {args.num_prompts} prompts):")
    print(f"  {'Language':<12} {'Epoch 1':>10} {'Epoch 2':>10} {'Change':>10} {'Prompts':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for _, row in df.iterrows():
        change = row['improvement']
        sign = '+' if change >= 0 else ''
        print(f"  {row['language']:12} {row['epoch1']:10.3f} {row['epoch2']:10.3f} "
              f"{sign}{change:9.3f} {row['count']:8.0f}")

    # Create grouped bar chart
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    languages = df['language'].values
    epoch1 = df['epoch1'].values
    epoch2 = df['epoch2'].values

    x = np.arange(len(languages))
    width = 0.35

    if args.disjoint:
        label1 = 'Early Training (First 50%, Epoch 1)'
        label2 = 'Late Training (Last 50%, Epoch 2)'
    else:
        label1 = 'Epoch 1 (First Exposure)'
        label2 = 'Epoch 2 (Second Exposure)'

    bars1 = ax.bar(x - width/2, epoch1, width, label=label1,
                   color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x + width/2, epoch2, width, label=label2,
                   color='#E74C3C', alpha=0.8)

    # Styling
    ax.set_xlabel('Programming Language', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=14, fontweight='bold')

    if args.disjoint:
        title = f'{run.name}: Early vs Late Training (First 50% vs Last 50% of Problems)'
    else:
        title = f'{run.name}: Reward on Same Prompts Across Epochs'

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
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
    suffix = "_disjoint" if args.disjoint else ""
    output_path = get_output_filename(
        f"language_reward_epochs_n{args.num_prompts}{suffix}",
        args.run_id,
        plot_type="analysis"
    )
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to: {output_path}.png")

    print(f"\n✅ Epoch comparison plot created successfully!")

if __name__ == "__main__":
    main()
