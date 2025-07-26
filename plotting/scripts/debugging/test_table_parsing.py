#!/usr/bin/env python3
"""
Test table parsing to extract rollout data from W&B runs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wandb_utils import get_run, get_table_data, parse_rollout_data


# Configuration - Using runs known to have table data
ENTITY = "assert-kth"
PROJECT = "SWE-Gym-GRPO"

# Test with different runs to find one with table data
test_runs = [
    "c1mr1lgd",  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/c1mr1lgd
    "bu2fqmm0",  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/bu2fqmm0
    "qa9t88ng",  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/qa9t88ng
]


def test_table_parsing():
    """Test parsing table data from various runs."""
    
    print("="*60)
    print("Testing Table Parsing for Rollout Data")
    print("="*60)
    
    for run_id in test_runs:
        print(f"\n{'='*40}")
        print(f"Testing run: {run_id}")
        print(f"{'='*40}")
        
        try:
            # Load run
            run = get_run(ENTITY, PROJECT, run_id)
            
            # Check if table exists
            if "table" in run.summary:
                print(f"✓ Table found in run {run.name}")
                table_info = run.summary["table"]
                print(f"Table info: {table_info}")
                
                # Parse table
                df = get_table_data(run)
                
                if df is not None:
                    print(f"\n📊 TABLE SUCCESSFULLY PARSED!")
                    print(f"Shape: {df.shape}")
                    print(f"Columns: {list(df.columns)}")
                    
                    # Show first few rows
                    print(f"\nFirst 3 rows:")
                    print(df.head(3).to_string())
                    
                    # Try to parse a rollout
                    print(f"\n🎯 PARSING FIRST ROLLOUT:")
                    rollout = parse_rollout_data(df, rollout_index=0)
                    
                    for key, value in rollout.items():
                        print(f"\n{key.upper()}:")
                        print("-" * 40)
                        if isinstance(value, str) and len(value) > 200:
                            print(f"{value[:200]}...")
                            print(f"[TRUNCATED - Full length: {len(value)} chars]")
                        else:
                            print(value)
                    
                    # Success - we found and parsed a table!
                    return True, run, df, rollout
                    
                else:
                    print("❌ Failed to parse table")
            else:
                print(f"❌ No table found in run {run.name}")
                
        except Exception as e:
            print(f"❌ Error with run {run_id}: {e}")
            continue
    
    print(f"\n❌ No tables found in any of the test runs")
    return False, None, None, None


def main():
    """Main function to test table parsing."""
    
    success, run, df, rollout = test_table_parsing()
    
    if success:
        print(f"\n🎉 SUCCESS! Parsed rollout data from {run.name}")
        print(f"Table shape: {df.shape}")
        print(f"Available rollouts: {len(df)}")
        
        # Print summary
        if 'prompt' in rollout and 'completion' in rollout:
            prompt_len = len(rollout['prompt']) if rollout['prompt'] else 0
            completion_len = len(rollout['completion']) if rollout['completion'] else 0
            print(f"\nRollout summary:")
            print(f"  Prompt length: {prompt_len} chars")
            print(f"  Completion length: {completion_len} chars")
            
            if 'reward' in rollout:
                print(f"  Reward: {rollout['reward']}")
    else:
        print(f"\n❌ Failed to find any table data in the test runs")
        print(f"You may need to check which runs actually contain table data")


if __name__ == "__main__":
    main()