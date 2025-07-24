#!/usr/bin/env python3
"""
Check all long runs for rm -rf patterns without CSV saving.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import tempfile
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from plot_config import ENTITY, PROJECT


def analyze_run_for_rm_patterns(run_id, run_name, sample_size=3):
    """
    Analyze a run for rm -rf patterns.
    """
    print(f"\nAnalyzing {run_name} ({run_id})")
    
    try:
        api = wandb.Api()
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        
        # Get HTML files
        files = run.files()
        html_files = [f for f in files if f.name.endswith('.html') and 'table' in f.name]
        
        print(f"  Found {len(html_files)} HTML files")
        
        if len(html_files) == 0:
            return {
                'run_id': run_id,
                'run_name': run_name,
                'status': 'no_html_files',
                'rollouts': 0,
                'commands': 0,
                'rm_rf_count': 0,
                'rm_rf_with_git': 0
            }
        
        # Analyze sample of files
        sample_files = html_files[:sample_size]
        print(f"  Analyzing {len(sample_files)} sample files...")
        
        total_rollouts = 0
        total_commands = 0
        rm_rf_instances = []
        
        # Define rm patterns
        rm_patterns = [
            r'rm\s+-rf\s+\S+',
            r'rm\s+-fr\s+\S+',
            r'rm\s+-r\s+-f\s+\S+',
            r'rm\s+-f\s+-r\s+\S+'
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in sample_files:
                try:
                    # Download and parse
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
                            
                            if len(completion_text) > 100:
                                total_rollouts += 1
                                
                                # Count shell commands
                                shell_count = len(re.findall(r'{"name"\s*:\s*"shell"', completion_text))
                                apply_count = len(re.findall(r'{"name"\s*:\s*"apply_patch"', completion_text))
                                total_commands += shell_count + apply_count
                                
                                # Check for rm -rf patterns
                                for pattern in rm_patterns:
                                    matches = re.findall(pattern, completion_text, re.IGNORECASE)
                                    if matches:
                                        has_git_clone = 'git clone' in completion_text
                                        
                                        for match in matches:
                                            rm_rf_instances.append({
                                                'file': file.name,
                                                'row': row_idx,
                                                'command': match,
                                                'has_git_clone': has_git_clone,
                                                'context_snippet': completion_text[:300] + '...'
                                            })
                
                except Exception as e:
                    print(f"    Error with {file.name}: {e}")
                    continue
        
        rm_rf_with_git = sum(1 for inst in rm_rf_instances if inst['has_git_clone'])
        
        print(f"  Results:")
        print(f"    Rollouts: {total_rollouts}")
        print(f"    Commands: {total_commands}")
        print(f"    rm -rf instances: {len(rm_rf_instances)}")
        print(f"    rm -rf with git: {rm_rf_with_git}")
        
        return {
            'run_id': run_id,
            'run_name': run_name,
            'status': 'success',
            'rollouts': total_rollouts,
            'commands': total_commands,
            'rm_rf_count': len(rm_rf_instances),
            'rm_rf_with_git': rm_rf_with_git,
            'instances': rm_rf_instances
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'run_id': run_id,
            'run_name': run_name,
            'status': 'error',
            'error': str(e),
            'rollouts': 0,
            'commands': 0,
            'rm_rf_count': 0,
            'rm_rf_with_git': 0
        }


def main():
    """Main function to check all long runs."""
    
    # The 6 long runs found
    long_runs = [
        ('nz1r7ml3', 'Qwen3-8B-3GPU-HighRank', 14.1),
        ('c1mr1lgd', 'Qwen3-8B-GRPO-GS4-0.01KL', 10.2),
        ('ar7sdpb2', 'Qwen3-8B-Thinking-Lora-Mask-Lite', 9.7),
        ('qa9t88ng', 'Qwen3-8B-Lora-NoMask-Lite', 6.9),
        ('bu2fqmm0', 'Qwen3-8B-Lora-Mask-Lite', 6.5),
        ('ytukh2p3', 'Qwen/Qwen3-8B-GRPO-swe_gym-repo_repair', 5.7)
    ]
    
    print("="*80)
    print("CHECKING ALL LONG RUNS FOR rm -rf PATTERNS")
    print("="*80)
    
    all_results = []
    
    for run_id, run_name, hours in long_runs:
        print(f"\n{'-'*60}")
        print(f"Processing {run_name} ({hours:.1f} hours)")
        print(f"{'-'*60}")
        
        result = analyze_run_for_rm_patterns(run_id, run_name)
        all_results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    total_rollouts = sum(r['rollouts'] for r in all_results)
    total_commands = sum(r['commands'] for r in all_results)
    total_rm_rf = sum(r['rm_rf_count'] for r in all_results)
    total_rm_rf_git = sum(r['rm_rf_with_git'] for r in all_results)
    
    print(f"\nOverall statistics:")
    print(f"  Total rollouts analyzed: {total_rollouts:,}")
    print(f"  Total commands found: {total_commands:,}")
    print(f"  Total rm -rf instances: {total_rm_rf}")
    print(f"  rm -rf followed by git clone: {total_rm_rf_git}")
    
    # Show any runs with rm -rf patterns
    runs_with_rm_rf = [r for r in all_results if r['rm_rf_count'] > 0]
    
    if runs_with_rm_rf:
        print(f"\nRuns with rm -rf patterns:")
        for result in runs_with_rm_rf:
            print(f"\n{result['run_name']} ({result['run_id']}):")
            print(f"  rm -rf instances: {result['rm_rf_count']}")
            print(f"  With git clone: {result['rm_rf_with_git']}")
            
            if 'instances' in result:
                print(f"  Examples:")
                for i, inst in enumerate(result['instances'][:3]):
                    print(f"    {i+1}. {inst['command']}")
                    print(f"       Git clone: {inst['has_git_clone']}")
                    print(f"       Context: {inst['context_snippet'][:150]}...")
    else:
        print(f"\n🎉 NO rm -rf PATTERNS FOUND in any of the 6 long runs!")
    
    print(f"\nAnalysis complete!")


if __name__ == "__main__":
    main()