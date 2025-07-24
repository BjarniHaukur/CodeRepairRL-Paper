#!/usr/bin/env python3
"""
Extract actual individual commands from WandB table data by downloading 
and parsing the HTML files containing the raw completion data.
"""

import sys
import os
import argparse
import tempfile
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

from wandb_utils import get_run, get_history, SHELL_PATTERN, APPLY_PATCH_PATTERN
from plot_config import ENTITY, PROJECT, RUN_ID, get_output_filename


def download_and_parse_table(run_id, table_index=None):
    """
    Download the actual HTML table data from WandB and parse it.
    """
    print(f"\nDownloading table data for run: {run_id}")
    
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    
    # Get all files from the run
    files = run.files()
    
    # Find HTML table files
    html_files = []
    for file in files:
        if file.name.endswith('.html') and 'table' in file.name:
            html_files.append(file)
    
    print(f"Found {len(html_files)} HTML table files")
    
    if len(html_files) == 0:
        print("No HTML table files found!")
        return []
    
    # Download and process each HTML file
    all_rollouts = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Downloading to temporary directory: {temp_dir}")
        
        for file in tqdm(html_files, desc="Processing HTML files"):  # Process all files
            try:
                # Download the file
                local_path = os.path.join(temp_dir, file.name.replace('/', '_'))
                file.download(root=temp_dir, replace=True)
                
                # Find the downloaded file
                downloaded_path = None
                for root, dirs, files_in_dir in os.walk(temp_dir):
                    for fname in files_in_dir:
                        if fname.endswith('.html') and 'table' in fname:
                            downloaded_path = os.path.join(root, fname)
                            break
                    if downloaded_path:
                        break
                
                if downloaded_path:
                    rollouts = parse_html_table(downloaded_path)
                    all_rollouts.extend(rollouts)
                else:
                    print(f"Could not find downloaded file for {file.name}")
                    
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                continue
        
        return all_rollouts


def parse_html_table(html_file):
    """
    Parse an HTML table file to extract rollout completion data.
    """
    with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    
    # Find the table
    table = soup.find('table')
    if not table:
        return []
    
    # Get headers
    headers = []
    header_row = table.find('tr')
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
    
    # Find completion column index
    completion_idx = None
    for i, header in enumerate(headers):
        if 'completion' in header.lower():
            completion_idx = i
            break
    
    if completion_idx is None:
        return []
    
    # Extract data rows
    rollouts = []
    rows = table.find_all('tr')[1:]  # Skip header row
    
    for row in rows:
        cells = row.find_all(['td', 'th'])
        if len(cells) > completion_idx:
            completion_cell = cells[completion_idx]
            completion_text = completion_cell.get_text(strip=True)
            
            if completion_text and len(completion_text) > 50:  # Filter out empty/short completions
                rollouts.append({
                    'completion': completion_text,
                    'file': os.path.basename(html_file)
                })
    
    return rollouts


def extract_commands_from_rollouts(rollouts):
    """
    Extract individual shell commands from rollout completion data.
    """
    print(f"\nExtracting commands from {len(rollouts)} rollouts...")
    
    all_commands = []
    rm_rf_instances = []
    
    for i, rollout in enumerate(tqdm(rollouts, desc="Processing rollouts")):
        completion = rollout['completion']
        
        # Extract shell commands
        shell_matches = SHELL_PATTERN.findall(completion)
        
        # Extract apply_patch commands
        apply_patch_matches = APPLY_PATCH_PATTERN.findall(completion)
        
        # Process shell commands
        for cmd in shell_matches:
            cmd = cmd.strip()
            if cmd:
                # Get base command (first word)
                base_cmd = cmd.split()[0] if cmd.split() else cmd
                
                all_commands.append({
                    'rollout_idx': i,
                    'file': rollout['file'],
                    'command_type': 'shell',
                    'base_command': base_cmd,
                    'full_command': cmd,
                    'completion_length': len(completion)
                })
                
                # Check for rm -rf patterns
                if any(pattern in cmd for pattern in ['rm -rf', 'rm -fr', 'rm -r -f']):
                    # Look for git clone in the same completion
                    has_git_clone = 'git clone' in completion
                    
                    # Extract context around rm command
                    rm_pos = completion.find(cmd)
                    context_start = max(0, rm_pos - 300)
                    context_end = min(len(completion), rm_pos + 500)
                    context = completion[context_start:context_end]
                    
                    rm_rf_instances.append({
                        'rollout_idx': i,
                        'file': rollout['file'],
                        'command': cmd,
                        'has_git_clone': has_git_clone,
                        'context': context
                    })
        
        # Process apply_patch commands
        for _ in apply_patch_matches:
            all_commands.append({
                'rollout_idx': i,
                'file': rollout['file'],
                'command_type': 'apply_patch',
                'base_command': 'apply_patch',
                'full_command': 'apply_patch',
                'completion_length': len(completion)
            })
    
    return all_commands, rm_rf_instances


def analyze_command_patterns(commands, rm_rf_instances):
    """
    Analyze the extracted command patterns.
    """
    print(f"\n{'='*80}")
    print("COMMAND ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nTotal commands extracted: {len(commands)}")
    print(f"Total rm -rf instances: {len(rm_rf_instances)}")
    
    # Command frequency analysis
    from collections import Counter
    cmd_counts = Counter(cmd['base_command'] for cmd in commands)
    
    print(f"\nTop 20 most frequent commands:")
    for cmd, count in cmd_counts.most_common(20):
        print(f"  {cmd}: {count}")
    
    # Check expected vs actual
    expected_rollouts = 150 * 8 * 8  # 9,600
    expected_commands = expected_rollouts * 15  # 144,000
    
    print(f"\nScale verification:")
    print(f"  Expected rollouts: {expected_rollouts:,}")
    print(f"  Actual rollouts: {len(set(cmd['rollout_idx'] for cmd in commands)):,}")
    print(f"  Expected commands: {expected_commands:,}")
    print(f"  Actual commands: {len(commands):,}")
    print(f"  Commands per rollout: {len(commands) / len(set(cmd['rollout_idx'] for cmd in commands)):.1f}")
    
    # rm -rf analysis
    if rm_rf_instances:
        print(f"\n{'='*60}")
        print("rm -rf EXPLOITATION ANALYSIS")
        print("="*60)
        
        with_git_clone = [inst for inst in rm_rf_instances if inst['has_git_clone']]
        
        print(f"\nTotal rm -rf instances: {len(rm_rf_instances)}")
        print(f"Instances with git clone: {len(with_git_clone)} ({len(with_git_clone)/len(rm_rf_instances)*100:.1f}%)")
        
        # Show examples
        print(f"\nExamples of rm -rf followed by git clone:")
        for i, inst in enumerate(with_git_clone[:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"File: {inst['file']}")
            print(f"Command: {inst['command']}")
            print(f"Context: {inst['context'][:300]}...")
        
        # Show examples without git clone
        without_git_clone = [inst for inst in rm_rf_instances if not inst['has_git_clone']]
        print(f"\nExamples of rm -rf without git clone:")
        for i, inst in enumerate(without_git_clone[:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"File: {inst['file']}")
            print(f"Command: {inst['command']}")
            print(f"Context: {inst['context'][:200]}...")
    
    # Save detailed results
    df_commands = pd.DataFrame(commands)
    df_rm_rf = pd.DataFrame(rm_rf_instances)
    
    cmd_output = get_output_filename('actual_commands_analysis', RUN_ID) + '.csv'
    df_commands.to_csv(cmd_output, index=False)
    print(f"\nDetailed command data saved to: {cmd_output}")
    
    if len(df_rm_rf) > 0:
        rm_output = get_output_filename('rm_rf_instances', RUN_ID) + '.csv'
        df_rm_rf.to_csv(rm_output, index=False)
        print(f"rm -rf instances saved to: {rm_output}")
    
    return df_commands, df_rm_rf


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Extract actual commands from WandB table data')
    parser.add_argument('--run-id', type=str, default=RUN_ID, 
                        help=f'WandB run ID (default: {RUN_ID})')
    args = parser.parse_args()
    
    # Download and parse table data
    rollouts = download_and_parse_table(args.run_id)
    
    if not rollouts:
        print("No rollout data found!")
        return
    
    # Extract commands
    commands, rm_rf_instances = extract_commands_from_rollouts(rollouts)
    
    # Analyze patterns
    df_commands, df_rm_rf = analyze_command_patterns(commands, rm_rf_instances)
    
    print(f"\nAnalysis complete!")


if __name__ == "__main__":
    main()