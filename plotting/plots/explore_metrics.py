#!/usr/bin/env python3
"""
Explore available metrics in the runs to find interesting data to plot.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wandb_utils import get_run, get_history, get_summary_metrics


# Configuration
ENTITY = "assert-kth"
PROJECT = "SWE-Gym-GRPO"
RUN1_ID = "c1mr1lgd"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/c1mr1lgd
RUN2_ID = "bu2fqmm0"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/bu2fqmm0


def main():
    """Explore metrics available in the runs."""
    
    print("="*60)
    print("Exploring Available Metrics")
    print("="*60)
    
    # Get runs
    run1 = get_run(ENTITY, PROJECT, RUN1_ID)
    run2 = get_run(ENTITY, PROJECT, RUN2_ID)
    
    # Get all available history (first few rows to see columns)
    print("\n" + "="*40)
    print(f"Run 1: {run1.name}")
    print("="*40)
    history1_sample = get_history(run1, keys=None)
    
    print("\n" + "="*40)
    print(f"Run 2: {run2.name}")
    print("="*40)
    history2_sample = get_history(run2, keys=None)
    
    # Get summary metrics
    print("\n" + "="*40)
    print(f"Summary Metrics - Run 1: {run1.name}")
    print("="*40)
    summary1 = get_summary_metrics(run1)
    
    print("\n" + "="*40)
    print(f"Summary Metrics - Run 2: {run2.name}")
    print("="*40)
    summary2 = get_summary_metrics(run2)


if __name__ == "__main__":
    main()