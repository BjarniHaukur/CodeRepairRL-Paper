#!/usr/bin/env python3
"""
Find all runs longer than 5 hours in the SWE-Gym-GRPO entity
and analyze them for rm -rf patterns.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

from plot_config import ENTITY, PROJECT, get_output_filename


def find_long_runs(min_hours=5):
    """
    Find all runs longer than min_hours in the entity.
    """
    print(f"Finding runs longer than {min_hours} hours in {ENTITY}/{PROJECT}")
    
    api = wandb.Api()
    
    # Get all runs from the project
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    long_runs = []
    
    for run in tqdm(runs, desc="Analyzing runs"):
        try:
            # Get runtime in seconds
            runtime = run.summary.get('_runtime', 0)
            
            if runtime is None:
                continue
                
            # Convert to hours
            runtime_hours = runtime / 3600
            
            if runtime_hours >= min_hours:
                # Get additional info
                steps = run.summary.get('_step', 0)
                created = run.created_at
                
                # Check if run has table data
                has_table_data = False
                try:
                    files = run.files()
                    html_files = [f for f in files if f.name.endswith('.html') and 'table' in f.name]
                    has_table_data = len(html_files) > 0
                except:
                    pass
                
                long_runs.append({
                    'run_id': run.id,
                    'run_name': run.name,
                    'state': run.state,
                    'runtime_hours': runtime_hours,
                    'runtime_seconds': runtime,
                    'steps': steps,
                    'created_at': created,
                    'has_table_data': has_table_data,
                    'url': run.url
                })
                
        except Exception as e:
            print(f"Error processing run {run.id}: {e}")
            continue
    
    return long_runs


def analyze_run_for_rm_patterns(run_info):
    """
    Analyze a specific run for rm -rf patterns.
    """
    run_id = run_info['run_id']
    
    print(f"\nAnalyzing {run_info['run_name']} ({run_id})")
    print(f"  Runtime: {run_info['runtime_hours']:.1f} hours")
    print(f"  Steps: {run_info['steps']}")
    print(f"  Has table data: {run_info['has_table_data']}")
    
    if not run_info['has_table_data']:
        print("  No table data - skipping")
        return {
            'run_id': run_id,
            'run_name': run_info['run_name'],
            'rollouts_processed': 0,
            'commands_found': 0,
            'rm_rf_instances': 0,
            'rm_rf_with_git': 0,
            'error': 'No table data'
        }
    
    try:
        # Get API and run
        api = wandb.Api()
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        
        # Get HTML files
        files = run.files()
        html_files = [f for f in files if f.name.endswith('.html') and 'table' in f.name]
        
        print(f"  Found {len(html_files)} HTML files")
        
        # Sample analysis - check first 5 files for patterns
        sample_files = html_files[:5]
        
        total_rollouts = 0
        total_commands = 0
        rm_rf_instances = []
        
        import tempfile
        from bs4 import BeautifulSoup
        import re
        
        # Define patterns
        rm_patterns = [
            r'rm\s+-rf\s+',
            r'rm\s+-fr\s+',
            r'rm\s+-r\s+-f\s+',
            r'rm\s+-f\s+-r\s+'
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in sample_files:
                try:
                    # Download file
                    file.download(root=temp_dir, replace=True)
                    
                    # Find downloaded file
                    downloaded_path = None
                    for root, dirs, files_in_dir in os.walk(temp_dir):
                        for fname in files_in_dir:
                            if fname.endswith('.html') and 'table' in fname:
                                downloaded_path = os.path.join(root, fname)
                                break
                        if downloaded_path:
                            break
                    
                    if not downloaded_path:
                        continue
                    
                    # Parse HTML
                    with open(downloaded_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    soup = BeautifulSoup(content, 'html.parser')
                    table = soup.find('table')
                    
                    if not table:
                        continue
                    
                    # Find completion column
                    headers = []
                    header_row = table.find('tr')
                    if header_row:
                        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                    
                    completion_idx = None
                    for i, header in enumerate(headers):
                        if 'completion' in header.lower():
                            completion_idx = i
                            break
                    
                    if completion_idx is None:
                        continue
                    
                    # Process rows
                    rows = table.find_all('tr')[1:]  # Skip header
                    
                    for row_idx, row in enumerate(rows):
                        cells = row.find_all(['td', 'th'])
                        if len(cells) > completion_idx:
                            completion_text = cells[completion_idx].get_text(strip=True)
                            
                            if len(completion_text) > 100:  # Filter short completions
                                total_rollouts += 1
                                
                                # Count commands (rough estimate)
                                shell_commands = len(re.findall(r'{"name"\s*:\s*"shell"', completion_text))
                                apply_patches = len(re.findall(r'{"name"\s*:\s*"apply_patch"', completion_text))
                                total_commands += shell_commands + apply_patches
                                
                                # Check for rm -rf patterns
                                for pattern in rm_patterns:
                                    matches = re.findall(pattern, completion_text, re.IGNORECASE)
                                    if matches:
                                        # Check if git clone follows
                                        has_git_clone = 'git clone' in completion_text
                                        
                                        rm_rf_instances.append({
                                            'file': file.name,
                                            'row': row_idx,
                                            'pattern': pattern,
                                            'matches': len(matches),
                                            'has_git_clone': has_git_clone,
                                            'context': completion_text[:500] + '...' if len(completion_text) > 500 else completion_text
                                        })
                
                except Exception as e:
                    print(f"    Error processing {file.name}: {e}")
                    continue
        
        rm_rf_with_git = sum(1 for inst in rm_rf_instances if inst['has_git_clone'])
        
        print(f"  Sample results:")
        print(f"    Rollouts processed: {total_rollouts}")
        print(f"    Commands found: {total_commands}")
        print(f"    rm -rf instances: {len(rm_rf_instances)}")
        print(f"    rm -rf with git clone: {rm_rf_with_git}")
        
        return {
            'run_id': run_id,
            'run_name': run_info['run_name'],
            'rollouts_processed': total_rollouts,
            'commands_found': total_commands,
            'rm_rf_instances': len(rm_rf_instances),
            'rm_rf_with_git': rm_rf_with_git,
            'instances_details': rm_rf_instances,
            'error': None
        }
        
    except Exception as e:
        return {
            'run_id': run_id,
            'run_name': run_info['run_name'],
            'rollouts_processed': 0,
            'commands_found': 0,
            'rm_rf_instances': 0,
            'rm_rf_with_git': 0,
            'error': str(e)
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Find and analyze long runs for rm -rf patterns')
    parser.add_argument('--min-hours', type=float, default=5.0, 
                        help='Minimum runtime in hours (default: 5.0)')
    parser.add_argument('--analyze', action='store_true', 
                        help='Analyze found runs for rm -rf patterns')
    args = parser.parse_args()
    
    # Find long runs
    long_runs = find_long_runs(args.min_hours)
    
    if not long_runs:
        print(f"No runs found longer than {args.min_hours} hours")
        return
    
    # Sort by runtime
    long_runs.sort(key=lambda x: x['runtime_hours'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"FOUND {len(long_runs)} RUNS LONGER THAN {args.min_hours} HOURS")
    print(f"{'='*80}")
    
    # Display summary
    for i, run in enumerate(long_runs, 1):
        print(f"\n{i}. {run['run_name']} ({run['run_id']})")
        print(f"   Runtime: {run['runtime_hours']:.1f} hours ({run['runtime_seconds']:.0f}s)")
        print(f"   Steps: {run['steps']:,}")
        print(f"   State: {run['state']}")
        print(f"   Created: {run['created_at']}")
        print(f"   Has table data: {run['has_table_data']}")
        print(f"   URL: {run['url']}")
    
    # Save summary
    df = pd.DataFrame(long_runs)
    summary_file = get_output_filename('long_runs_summary', 'all_runs') + '.csv'
    df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
    
    # Analyze if requested
    if args.analyze:
        print(f"\n{'='*80}")
        print("ANALYZING RUNS FOR rm -rf PATTERNS")
        print(f"{'='*80}")
        
        analysis_results = []
        
        for run_info in tqdm(long_runs, desc="Analyzing runs"):
            result = analyze_run_for_rm_patterns(run_info)
            analysis_results.append(result)
        
        # Summary of analysis
        print(f"\n{'='*80}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        total_rollouts = sum(r['rollouts_processed'] for r in analysis_results)
        total_commands = sum(r['commands_found'] for r in analysis_results)
        total_rm_rf = sum(r['rm_rf_instances'] for r in analysis_results)
        total_rm_rf_git = sum(r['rm_rf_with_git'] for r in analysis_results)
        
        print(f"\nOverall results:")
        print(f"  Total rollouts processed: {total_rollouts:,}")
        print(f"  Total commands found: {total_commands:,}")
        print(f"  Total rm -rf instances: {total_rm_rf}")
        print(f"  rm -rf with git clone: {total_rm_rf_git}")
        
        # Show runs with rm -rf patterns
        runs_with_patterns = [r for r in analysis_results if r['rm_rf_instances'] > 0]
        
        if runs_with_patterns:
            print(f"\nRuns with rm -rf patterns:")
            for result in runs_with_patterns:
                print(f"\n{result['run_name']} ({result['run_id']}):")
                print(f"  rm -rf instances: {result['rm_rf_instances']}")
                print(f"  With git clone: {result['rm_rf_with_git']}")
                
                # Show examples
                if 'instances_details' in result and result['instances_details']:
                    print(f"  Examples:")
                    for inst in result['instances_details'][:2]:
                        print(f"    File: {inst['file']}, Pattern: {inst['pattern']}")
                        print(f"    Has git clone: {inst['has_git_clone']}")
                        print(f"    Context: {inst['context'][:200]}...")
        else:
            print(f"\nNo rm -rf patterns found in any of the analyzed runs!")
        
        # Save analysis results
        df_analysis = pd.DataFrame(analysis_results)
        analysis_file = get_output_filename('rm_rf_analysis_all_runs', 'all_runs') + '.csv'
        df_analysis.to_csv(analysis_file, index=False)
        print(f"\nAnalysis results saved to: {analysis_file}")


if __name__ == "__main__":
    main()