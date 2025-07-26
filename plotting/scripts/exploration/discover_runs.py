#!/usr/bin/env python3
"""
Discover and analyze all runs in a project to identify viable ones for plotting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wandb_utils import get_all_runs, analyze_runs


# Configuration
ENTITY = "assert-kth"
PROJECT = "SWE-Gym-GRPO"


def main():
    """Discover and analyze all runs in the project."""
    
    print("="*60)
    print("Run Discovery and Analysis")
    print("="*60)
    
    # Get all runs
    runs = get_all_runs(ENTITY, PROJECT)
    
    # Analyze runs for viability
    analysis = analyze_runs(runs, min_steps=100)
    
    # Print viable runs with URLs for easy copying
    print("\n" + "="*60)
    print("VIABLE RUNS FOR PLOTTING:")
    print("="*60)
    
    for i, run_info in enumerate(analysis["viable_runs"]):
        run = run_info["run"]
        steps = run_info["steps"]
        url = run_info["url"]
        
        print(f"\n{i+1}. {run.name}")
        print(f"   ID: {run.id}")
        print(f"   Steps: {steps:.0f}")
        print(f"   State: {run.state}")
        print(f"   Tags: {run.tags}")
        print(f"   URL: {url}")
        print(f"   Copy-paste line:")
        print(f'   run_{i+1} = "{run.id}"  # {url}')
    
    # Print interesting crashed runs (might still be worth plotting)
    print("\n" + "="*60)
    print("CRASHED RUNS WITH >100 STEPS (might be interesting):")
    print("="*60)
    
    crashed_viable = [r for r in analysis["crashed_runs"] if r["steps"] >= 100]
    for i, run_info in enumerate(crashed_viable):
        run = run_info["run"]
        steps = run_info["steps"]
        url = run_info["url"]
        
        print(f"\n{i+1}. {run.name} (CRASHED)")
        print(f"   ID: {run.id}")
        print(f"   Steps: {steps:.0f}")
        print(f"   Tags: {run.tags}")
        print(f"   URL: {url}")
        print(f"   Copy-paste line:")
        print(f'   crashed_run_{i+1} = "{run.id}"  # {url}')
    
    # Generate example plot configuration
    print("\n" + "="*60)
    print("EXAMPLE PLOT CONFIGURATION:")
    print("="*60)
    
    viable_runs = analysis["viable_runs"][:4]  # Take first 4 viable runs
    
    print("\n# Example configuration for a new plot script:")
    print("# Copy these lines to your plots/your_plot.py file")
    print()
    
    for i, run_info in enumerate(viable_runs):
        run = run_info["run"]
        url = run_info["url"]
        var_name = f"run_{chr(97+i)}"  # run_a, run_b, run_c, run_d
        print(f'{var_name} = "{run.id}"  # {url}')
    
    print()
    print("# Or as a list:")
    run_ids = [f'"{r["run"].id}"' for r in viable_runs]
    urls = [f'# {r["url"]}' for r in viable_runs]
    print("runs = [")
    for run_id, url in zip(run_ids, urls):
        print(f"    {run_id},  {url}")
    print("]")


if __name__ == "__main__":
    main()