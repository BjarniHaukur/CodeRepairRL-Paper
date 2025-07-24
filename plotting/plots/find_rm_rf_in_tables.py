#!/usr/bin/env python3
"""
Find and analyze rm -rf patterns by extracting actual table data from history.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import pandas as pd
import json
import re
from tqdm import tqdm
from wandb_utils import get_run, get_history
from plot_config import ENTITY, PROJECT, RUN_ID


def extract_table_data(run_id):
    """
    Extract actual table data from run history.
    """
    print(f"\nExtracting table data for run: {run_id}")
    
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    
    # Download full run data
    print("Downloading run data...")
    history_path = run.download()
    
    # Look for parquet files in the download
    import glob
    parquet_files = glob.glob(os.path.join(history_path, "**/*.parquet"), recursive=True)
    json_files = glob.glob(os.path.join(history_path, "**/*.json"), recursive=True)
    
    print(f"Found {len(parquet_files)} parquet files and {len(json_files)} json files")
    
    rm_rf_instances = []
    
    # Process parquet files
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            print(f"\nProcessing {pf}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)[:10]}")
            
            # Look for completion data
            for col in df.columns:
                if 'completion' in col.lower() or 'response' in col.lower() or 'output' in col.lower():
                    print(f"Checking column: {col}")
                    for idx, value in df[col].items():
                        if pd.notna(value) and isinstance(value, str):
                            if 'rm -rf' in value or 'rm -fr' in value:
                                # Extract context
                                rm_pos = value.find('rm -rf')
                                if rm_pos == -1:
                                    rm_pos = value.find('rm -fr')
                                
                                context_start = max(0, rm_pos - 300)
                                context_end = min(len(value), rm_pos + 500)
                                context = value[context_start:context_end]
                                
                                # Check for git clone
                                has_git_clone = 'git clone' in value[rm_pos:rm_pos+1000]
                                
                                rm_rf_instances.append({
                                    'file': pf,
                                    'row': idx,
                                    'column': col,
                                    'context': context,
                                    'has_git_clone': has_git_clone,
                                    'full_text_length': len(value)
                                })
        except Exception as e:
            print(f"Error processing {pf}: {e}")
    
    # Process JSON files
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            
            print(f"\nProcessing {jf}")
            
            # Recursively search for completion-like fields
            def search_for_completions(obj, path=""):
                if isinstance(obj, str) and len(obj) > 100:
                    if 'rm -rf' in obj or 'rm -fr' in obj:
                        rm_pos = obj.find('rm -rf')
                        if rm_pos == -1:
                            rm_pos = obj.find('rm -fr')
                        
                        context_start = max(0, rm_pos - 300)
                        context_end = min(len(obj), rm_pos + 500)
                        context = obj[context_start:context_end]
                        
                        has_git_clone = 'git clone' in obj[rm_pos:rm_pos+1000]
                        
                        rm_rf_instances.append({
                            'file': jf,
                            'path': path,
                            'context': context,
                            'has_git_clone': has_git_clone,
                            'full_text_length': len(obj)
                        })
                
                elif isinstance(obj, dict):
                    for key, value in obj.items():
                        search_for_completions(value, f"{path}/{key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        search_for_completions(item, f"{path}[{i}]")
            
            search_for_completions(data)
            
        except Exception as e:
            print(f"Error processing {jf}: {e}")
    
    return rm_rf_instances


def analyze_rm_rf_patterns(instances):
    """
    Analyze the found rm -rf patterns.
    """
    print(f"\n{'='*80}")
    print(f"ANALYSIS OF rm -rf PATTERNS")
    print(f"{'='*80}")
    
    print(f"\nTotal instances found: {len(instances)}")
    
    with_git_clone = [i for i in instances if i['has_git_clone']]
    print(f"Instances followed by git clone: {len(with_git_clone)} ({len(with_git_clone)/len(instances)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("EXAMPLES OF rm -rf FOLLOWED BY git clone")
    print("="*60)
    
    for i, instance in enumerate(with_git_clone[:5]):
        print(f"\n--- Example {i+1} ---")
        print(f"File: {instance['file'].split('/')[-1] if 'file' in instance else 'unknown'}")
        if 'row' in instance:
            print(f"Location: Row {instance['row']}, Column {instance['column']}")
        elif 'path' in instance:
            print(f"JSON Path: {instance['path']}")
        print(f"Context ({len(instance['context'])} chars):")
        print("-" * 40)
        print(instance['context'])
        print("-" * 40)
    
    print("\n" + "="*60)
    print("EXAMPLES OF rm -rf WITHOUT git clone")
    print("="*60)
    
    without_git_clone = [i for i in instances if not i['has_git_clone']]
    for i, instance in enumerate(without_git_clone[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"File: {instance['file'].split('/')[-1] if 'file' in instance else 'unknown'}")
        print(f"Context snippet:")
        print(instance['context'][:300] + "...")


def main():
    """Main function."""
    # Analyze specific run
    instances = extract_table_data(RUN_ID)
    
    if instances:
        analyze_rm_rf_patterns(instances)
    else:
        print("\nNo rm -rf patterns found in this run!")
        
    # Try other runs with more activity
    print("\n" + "="*80)
    print("Checking other runs for rm -rf patterns...")
    print("="*80)
    
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"state": "finished"})
    
    # Check runs with longer training
    for run in list(runs)[:5]:
        if run.summary.get('_step', 0) > 1000:
            print(f"\nChecking {run.name} ({run.id}) with {run.summary.get('_step', 0)} steps...")
            instances = extract_table_data(run.id)
            if instances:
                print(f"Found {len(instances)} rm -rf instances!")
                with_clone = sum(1 for i in instances if i['has_git_clone'])
                print(f"  {with_clone} followed by git clone")


if __name__ == "__main__":
    main()