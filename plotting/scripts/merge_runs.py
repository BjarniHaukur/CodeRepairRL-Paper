#!/usr/bin/env python3
"""
Merge two continued training runs into a single unified history.

When a training run crashes and is resumed, this utility merges the two runs
by having the continuation run backpopulate over the crashed run's steps.

Usage:
    # Standalone
    uv run -m scripts.merge_runs 8dc73bp4 jb5uxlqc

    # As module in other scripts
    from scripts.merge_runs import merge_continued_runs
    merged_hist, run, _ = merge_continued_runs('8dc73bp4', 'jb5uxlqc')
"""

import argparse
import pandas as pd
from wandb_utils import get_run
from plot_config import ENTITY, PROJECT


def merge_continued_runs(
    run_id_1: str,
    run_id_2: str,
    entity: str = ENTITY,
    project: str = PROJECT
):
    """
    Merge two continued training runs.

    The run with the higher max step count is assumed to be the continuation.
    Its data will backpopulate over the earlier run's data where they overlap.

    Args:
        run_id_1: First run ID
        run_id_2: Second run ID
        entity: WandB entity
        project: WandB project

    Returns:
        Tuple of (merged_history, primary_run, earlier_run)
            - merged_history: pandas DataFrame with merged data
            - primary_run: wandb Run object for the continuation run
            - earlier_run: wandb Run object for the earlier run
    """
    print("="*60)
    print("Merging Continued Training Runs")
    print("="*60)

    # Get both runs
    print(f"\nFetching run 1: {run_id_1}")
    run1 = get_run(entity, project, run_id_1)

    print(f"\nFetching run 2: {run_id_2}")
    run2 = get_run(entity, project, run_id_2)

    # Get full history for both runs (fetch ALL data, not just 500 samples)
    print("\nFetching run 1 history (all data)...")
    hist1 = run1.history(samples=100000)  # Fetch up to 100k samples
    print(f"  Shape: {hist1.shape}")

    print("Fetching run 2 history (all data)...")
    hist2 = run2.history(samples=100000)  # Fetch up to 100k samples
    print(f"  Shape: {hist2.shape}")

    # Determine which is the continuation based on max step
    if '_step' not in hist1.columns or '_step' not in hist2.columns:
        raise ValueError("Both runs must have '_step' column in history")

    max_step_1 = hist1['_step'].max() if len(hist1) > 0 else 0
    max_step_2 = hist2['_step'].max() if len(hist2) > 0 else 0

    min_step_1 = hist1['_step'].min() if len(hist1) > 0 else 0
    min_step_2 = hist2['_step'].min() if len(hist2) > 0 else 0

    print(f"\nRun 1 ({run_id_1}):")
    print(f"  Name: {run1.name}")
    print(f"  Step range: {min_step_1} - {max_step_1}")

    print(f"\nRun 2 ({run_id_2}):")
    print(f"  Name: {run2.name}")
    print(f"  Step range: {min_step_2} - {max_step_2}")

    # The continuation is the one with higher max step
    if max_step_2 > max_step_1:
        earlier_run, earlier_hist = run1, hist1
        continuation_run, continuation_hist = run2, hist2
        earlier_id, continuation_id = run_id_1, run_id_2
        print(f"\n✓ Run 2 ({run_id_2}) is the continuation (higher max step)")
    else:
        earlier_run, earlier_hist = run2, hist2
        continuation_run, continuation_hist = run1, hist1
        earlier_id, continuation_id = run_id_2, run_id_1
        print(f"\n✓ Run 1 ({run_id_1}) is the continuation (higher max step)")

    # Determine offset for continuation run
    # The continuation run restarts at step 0, but should continue from where earlier run ended
    max_earlier_step = earlier_hist['_step'].max() if len(earlier_hist) > 0 else 0
    min_continuation_step = continuation_hist['_step'].min() if len(continuation_hist) > 0 else 0

    print(f"\nMerge analysis:")
    print(f"  Earlier run: steps 0-{max_earlier_step:.0f} ({len(earlier_hist)} rows)")
    print(f"  Continuation run (raw): steps {min_continuation_step:.0f}-{continuation_hist['_step'].max():.0f} ({len(continuation_hist)} rows)")

    # Calculate offset: continuation should start at max_earlier_step + 1
    step_offset = max_earlier_step + 1 - min_continuation_step
    print(f"  → Offsetting continuation steps by +{step_offset:.0f}")

    # Apply offset to continuation run
    continuation_hist_offset = continuation_hist.copy()
    continuation_hist_offset['_step'] = continuation_hist_offset['_step'] + step_offset

    new_max_step = continuation_hist_offset['_step'].max()
    print(f"  Continuation run (offset): steps {max_earlier_step + 1:.0f}-{new_max_step:.0f}")

    # Merge strategy: take all of earlier run, then all of continuation with offset
    merged = pd.concat([earlier_hist, continuation_hist_offset], ignore_index=True)

    # Sort by step to ensure proper ordering
    merged = merged.sort_values('_step').reset_index(drop=True)

    print(f"\n" + "="*60)
    print("Merge Results")
    print("="*60)
    print(f"Earlier run ({earlier_id}) contributed: {len(earlier_hist)} rows")
    print(f"Continuation run ({continuation_id}) contributed: {len(continuation_hist_offset)} rows")
    print(f"Total merged rows: {len(merged)}")
    print(f"Step range: {merged['_step'].min():.0f} - {merged['_step'].max():.0f}")
    print(f"Columns: {len(merged.columns)}")
    print(f"\n✓ Successfully merged continued training runs!")

    return merged, continuation_run, earlier_run


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Merge two continued training runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge two runs
  python -m scripts.merge_runs 8dc73bp4 jb5uxlqc

  # Merge with custom entity/project
  python -m scripts.merge_runs run1 run2 --entity my-entity --project my-project
        """
    )
    parser.add_argument('run_id_1', type=str, help='First run ID')
    parser.add_argument('run_id_2', type=str, help='Second run ID')
    parser.add_argument('--entity', type=str, default=ENTITY,
                        help=f'WandB entity (default: {ENTITY})')
    parser.add_argument('--project', type=str, default=PROJECT,
                        help=f'WandB project (default: {PROJECT})')
    args = parser.parse_args()

    # Merge the runs
    merged_history, primary_run, earlier_run = merge_continued_runs(
        args.run_id_1,
        args.run_id_2,
        args.entity,
        args.project
    )

    print(f"\n" + "="*60)
    print("✅ Successfully merged runs!")
    print("="*60)
    print(f"\nPrimary run (continuation): {primary_run.name}")
    print(f"  ID: {primary_run.id}")
    print(f"  URL: https://wandb.ai/{args.entity}/{args.project}/runs/{primary_run.id}")

    print(f"\nTo use this merged data in plotting scripts:")
    print(f"  1. Import: from scripts.merge_runs import merge_continued_runs")
    print(f"  2. Merge: merged_hist, run, _ = merge_continued_runs('{args.run_id_1}', '{args.run_id_2}')")
    print(f"  3. Use merged_hist in place of get_history(run, keys=...)")
    print(f"\nExample:")
    print(f"  # Instead of:")
    print(f"  #   history = get_history(run, keys=['_step', 'train/loss'])")
    print(f"  # Use:")
    print(f"  #   merged_hist, run, _ = merge_continued_runs('{args.run_id_1}', '{args.run_id_2}')")
    print(f"  #   history = merged_hist[['_step', 'train/loss']].dropna()")


if __name__ == "__main__":
    main()
