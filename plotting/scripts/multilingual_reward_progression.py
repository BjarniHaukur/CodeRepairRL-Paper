#!/usr/bin/env python3
"""
Plot reward progression across different programming languages (RQ3: Multilingual Training).

This script analyzes how the agent's performance varies across different programming
languages during training, showing reward trajectories for each language separately.

Run ID: 6wkkt1s0 (https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/6wkkt1s0)
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from wandb_utils import get_run
from plot_config import create_figure, save_figure, format_axis_labels, get_color
from utils.table_parser import TableExtractor


# Configuration
ENTITY = "assert-kth"
PROJECT = "SWE-Gym-GRPO"
RUN_ID = "6wkkt1s0"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/6wkkt1s0

# Language file extension patterns
LANGUAGE_PATTERNS = {
    'C': [r'\.c$', r'\.h$'],
    'C++': [r'\.cpp$', r'\.cc$', r'\.cxx$', r'\.hpp$', r'\.hh$', r'\.hxx$'],
    'Go': [r'\.go$'],
    'Java': [r'\.java$'],
    'JavaScript': [r'\.js$', r'\.jsx$', r'\.mjs$'],
    'TypeScript': [r'\.ts$', r'\.tsx$'],
    'PHP': [r'\.php$'],
    'Ruby': [r'\.rb$'],
    'Rust': [r'\.rs$'],
    'Python': [r'\.py$']  # Added Python since it's likely present
}

# Colors for each language (ensuring 10 distinguishable colors)
LANGUAGE_COLORS = {
    'C': '#E74C3C',          # Red
    'C++': '#9B59B6',        # Purple
    'Go': '#00ADD8',         # Go blue
    'Java': '#F89820',       # Java orange
    'JavaScript': '#F7DF1E', # JS yellow
    'TypeScript': '#3178C6', # TS blue
    'PHP': '#777BB4',        # PHP purple
    'Ruby': '#CC342D',       # Ruby red
    'Rust': '#CE422B',       # Rust orange
    'Python': '#3776AB'      # Python blue
}


def detect_language_from_patch(patch: str) -> str:
    """
    Detect programming language from git diff patch by examining file extensions.

    Args:
        patch: Git diff patch string

    Returns:
        Detected language name, or 'Unknown' if no match
    """
    # Extract filenames from diff headers
    # Look for patterns like: diff --git a/file.ext b/file.ext
    # or +++ b/file.ext, --- a/file.ext
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

    return 'Unknown'


def apply_ema_smoothing(values, alpha=0.01):
    """
    Apply exponential moving average smoothing.

    Args:
        values: Array of values to smooth
        alpha: Smoothing parameter (0 < alpha < 1, lower = more smoothing)

    Returns:
        Tuple of (smoothed array with first 0.5% removed, skip_points)
    """
    if len(values) == 0:
        return np.array([]), 0

    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]

    # Skip first 0.5% to avoid startup bias
    skip_points = max(1, int(len(smoothed) * 0.005))
    return smoothed[skip_points:], skip_points


def main():
    """Plot multilingual reward progression over training."""
    parser = argparse.ArgumentParser(description='Plot reward progression by programming language')
    parser.add_argument('--run-id', type=str, default=RUN_ID,
                        help=f'WandB run ID (default: {RUN_ID})')
    parser.add_argument('--ema-alpha', type=float, default=0.01,
                        help='EMA smoothing parameter (default: 0.01, higher = less smoothing)')
    parser.add_argument('--min-samples', type=int, default=10,
                        help='Minimum samples required to plot a language (default: 10)')
    parser.add_argument('--max-y', type=float, default=0.3,
                        help='Maximum y-axis value (default: 0.3)')
    args = parser.parse_args()

    print("="*60)
    print("Multilingual Reward Progression Analysis (RQ3)")
    print("="*60)

    # Get run and extract tables
    print(f"\n1. Loading run {args.run_id}...")
    run = get_run(ENTITY, PROJECT, args.run_id)
    print(f"   Run name: {run.name}")

    print(f"\n2. Extracting rollout tables...")
    extractor = TableExtractor()
    tables = extractor.extract_all_training_tables(run)

    if not tables:
        print("❌ No tables found")
        return

    # Process tables to extract language-specific rollouts
    print(f"\n3. Detecting languages from patches...")
    raw_data = []
    total_rollouts = 0
    language_counts = defaultdict(int)

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
                language_counts[language] += 1

                # Store data point
                raw_data.append({
                    'step': global_step,
                    'language': language,
                    'reward': reward
                })

                total_rollouts += 1

            except Exception as e:
                continue

    print(f"   Processed {total_rollouts} rollouts")
    print(f"\n   Language distribution:")
    for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"     {lang}: {count} rollouts ({100*count/total_rollouts:.1f}%)")

    # Aggregate data: group by (training_step, language) and calculate mean reward
    print(f"\n4. Aggregating rewards by training step and language...")
    df_raw = pd.DataFrame(raw_data)

    # Group by step and language, calculate mean reward
    df_aggregated = df_raw.groupby(['step', 'language'], as_index=False).agg({
        'reward': ['mean', 'count']
    })
    df_aggregated.columns = ['step', 'language', 'reward', 'count']

    print(f"   Aggregated {len(raw_data)} individual rollouts into {len(df_aggregated)} (step, language) pairs")

    # Convert back to dictionary structure for filtering
    language_data = defaultdict(list)
    for _, row in df_aggregated.iterrows():
        language_data[row['language']].append({
            'step': row['step'],
            'reward': row['reward'],
            'count': row['count']  # Track how many rollouts were averaged
        })

    # Filter languages by minimum sample size
    filtered_languages = {
        lang: data for lang, data in language_data.items()
        if len(data) >= args.min_samples and lang != 'Unknown'
    }

    if not filtered_languages:
        print(f"\n❌ No languages with at least {args.min_samples} samples")
        return

    print(f"\n5. Plotting {len(filtered_languages)} languages with >={args.min_samples} samples...")

    # Create plot
    fig, ax = create_figure(size="large")

    # Sort languages by total number of samples for consistent legend order
    sorted_languages = sorted(
        filtered_languages.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    # Plot each language
    for lang, data in sorted_languages:
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data).sort_values('step')

        steps = df['step'].values
        rewards = df['reward'].values
        total_rollouts_lang = sum(d['count'] for d in data)

        # Get color for this language
        color = LANGUAGE_COLORS.get(lang, '#95A5A6')

        # Apply EMA smoothing
        rewards_smooth, skip_points = apply_ema_smoothing(rewards, alpha=args.ema_alpha)
        steps_smooth = steps[skip_points:]
        ax.plot(steps_smooth, rewards_smooth,
               color=color,
               linewidth=2.5,
               alpha=0.85,
               label=f'{lang} ({len(data)} steps, {total_rollouts_lang} rollouts)')

    # Format axes
    format_axis_labels(ax,
                       xlabel='Training Step',
                       ylabel='Reward (Unified Diff Similarity)',
                       title=f'{run.name}: Multilingual Reward Progression')

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Place legend outside plot area on the right
    ax.legend(bbox_to_anchor=(1.05, 1),
             loc='upper left',
             fontsize=9,
             frameon=True,
             fancybox=True,
             shadow=True)

    # Set y-axis limits with margin
    ax.set_ylim(-0.05 * args.max_y, args.max_y * 1.05)

    # Add statistics box
    avg_rewards = {
        lang: np.mean([d['reward'] for d in data])
        for lang, data in sorted_languages
    }
    best_lang = max(avg_rewards.items(), key=lambda x: x[1])

    stats_text = (f'Languages: {len(filtered_languages)}\n'
                  f'Total Rollouts: {total_rollouts}\n'
                  f'Best Avg: {best_lang[0]} ({best_lang[1]:.3f})')

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            fontsize=10)

    # Save figure
    print(f"\n6. Saving figure...")
    output_filename = f"multilingual_rewards_{args.run_id}"
    save_figure(fig, output_filename, plot_type="temporal")

    print(f"\n✅ Multilingual reward progression plot created successfully!")
    print(f"   Output: figures/plots/temporal/{output_filename}.png")

    # Print per-language statistics
    print(f"\n📊 Per-Language Statistics (aggregated by training step):")
    for lang, data in sorted_languages:
        rewards = [d['reward'] for d in data]
        total_rollouts_lang = sum(d['count'] for d in data)
        print(f"\n   {lang}:")
        print(f"     Training steps: {len(rewards)}")
        print(f"     Total rollouts: {total_rollouts_lang}")
        print(f"     Avg rollouts/step: {total_rollouts_lang/len(rewards):.1f}")
        print(f"     Mean reward: {np.mean(rewards):.4f}")
        print(f"     Std reward: {np.std(rewards):.4f}")
        print(f"     Max reward: {np.max(rewards):.4f}")
        print(f"     Min reward: {np.min(rewards):.4f}")

    plt.close()


if __name__ == "__main__":
    main()
