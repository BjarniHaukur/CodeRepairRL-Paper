#!/usr/bin/env python3
"""
Comprehensive analysis of ALL tables in run nz1r7ml3 for rm -rf patterns.
Check every single rollout to see if rm -rf patterns emerge later in training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import tempfile
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import json
from plot_config import ENTITY, PROJECT, RUN_ID


def analyze_single_html_file(file, temp_dir, file_index):
    """
    Download and analyze a single HTML file for rm -rf patterns.
    """
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
            return None
        
        # Parse HTML
        with open(downloaded_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return None
        
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
            return None
        
        # Process all rows
        rows = table.find_all('tr')[1:]  # Skip header
        
        file_results = {
            'file_index': file_index,
            'file_name': file.name,
            'total_rows': len(rows),
            'rollouts': [],
            'rm_rf_instances': []
        }
        
        # Define rm patterns (more comprehensive)
        rm_patterns = [
            r'rm\s+-rf\s+\S+',           # rm -rf path
            r'rm\s+-fr\s+\S+',           # rm -fr path  
            r'rm\s+-r\s+-f\s+\S+',       # rm -r -f path
            r'rm\s+-f\s+-r\s+\S+',       # rm -f -r path
            r'rm\s+--recursive\s+--force\s+\S+',  # long form
            r'rm\s+-rf\s+\*',            # rm -rf *
            r'rm\s+-rf\s+\.',            # rm -rf .
            r'rm\s+-rf\s+/',             # rm -rf /
            r'rm\s+-rf\s+\w+',           # rm -rf word
        ]
        
        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            if len(cells) > completion_idx:
                completion_text = cells[completion_idx].get_text(strip=True)
                
                if len(completion_text) > 50:  # Filter very short completions
                    rollout_data = {
                        'row_index': row_idx,
                        'completion_length': len(completion_text),
                        'shell_commands': 0,
                        'apply_patches': 0,
                        'has_rm_rf': False,
                        'rm_rf_commands': []
                    }
                    
                    # Count commands
                    shell_count = len(re.findall(r'{"name"\s*:\s*"shell"', completion_text))
                    apply_count = len(re.findall(r'{"name"\s*:\s*"apply_patch"', completion_text))
                    rollout_data['shell_commands'] = shell_count
                    rollout_data['apply_patches'] = apply_count
                    
                    # Check for rm -rf patterns
                    for pattern in rm_patterns:
                        matches = re.findall(pattern, completion_text, re.IGNORECASE)
                        if matches:
                            rollout_data['has_rm_rf'] = True
                            
                            for match in matches:
                                # Check for git clone in same completion
                                has_git_clone = 'git clone' in completion_text
                                
                                # Get broader context around rm command
                                rm_pos = completion_text.find(match)
                                context_start = max(0, rm_pos - 500)
                                context_end = min(len(completion_text), rm_pos + 1000)
                                context = completion_text[context_start:context_end]
                                
                                rm_instance = {
                                    'file_index': file_index,
                                    'file_name': file.name,
                                    'row_index': row_idx,
                                    'pattern': pattern,
                                    'command': match,
                                    'has_git_clone': has_git_clone,
                                    'context': context,
                                    'completion_length': len(completion_text)
                                }
                                
                                rollout_data['rm_rf_commands'].append(rm_instance)
                                file_results['rm_rf_instances'].append(rm_instance)
                    
                    file_results['rollouts'].append(rollout_data)
        
        # Clean up downloaded file
        if downloaded_path and os.path.exists(downloaded_path):
            os.remove(downloaded_path)
        
        return file_results
        
    except Exception as e:
        return {
            'file_index': file_index,
            'file_name': file.name,
            'error': str(e),
            'total_rows': 0,
            'rollouts': [],
            'rm_rf_instances': []
        }


def comprehensive_analysis():
    """
    Analyze ALL HTML files in run nz1r7ml3 for rm -rf patterns.
    """
    print("="*80)
    print(f"COMPREHENSIVE rm -rf ANALYSIS FOR RUN: {RUN_ID}")
    print("="*80)
    
    # Get API with increased timeout
    api = wandb.Api(timeout=60)
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    
    print(f"Run: {run.name}")
    print(f"State: {run.state}")
    
    # Get ALL HTML files
    files = run.files()
    html_files = [f for f in files if f.name.endswith('.html') and 'table' in f.name]
    
    print(f"\nFound {len(html_files)} HTML table files")
    print("Processing ALL files (this will take a while)...")
    
    all_results = []
    total_rollouts = 0
    total_rm_rf = 0
    files_with_rm_rf = 0
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_index, file in enumerate(tqdm(html_files, desc="Processing HTML files")):
            result = analyze_single_html_file(file, temp_dir, file_index)
            
            if result:
                all_results.append(result)
                total_rollouts += len(result['rollouts'])
                if result['rm_rf_instances']:
                    total_rm_rf += len(result['rm_rf_instances'])
                    files_with_rm_rf += 1
                
                # Print progress every 25 files
                if (file_index + 1) % 25 == 0:
                    print(f"\nProgress: {file_index + 1}/{len(html_files)} files")
                    print(f"  Rollouts so far: {total_rollouts:,}")
                    print(f"  rm -rf instances so far: {total_rm_rf}")
    
    return all_results, total_rollouts, total_rm_rf, files_with_rm_rf


def analyze_results(all_results):
    """
    Analyze the comprehensive results.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("="*80)
    
    # Overall statistics
    total_files = len(all_results)
    total_rollouts = sum(len(r['rollouts']) for r in all_results)
    total_rm_rf = sum(len(r['rm_rf_instances']) for r in all_results)
    files_with_rm_rf = sum(1 for r in all_results if r['rm_rf_instances'])
    
    print(f"\nOverall Statistics:")
    print(f"  Files processed: {total_files}")
    print(f"  Total rollouts: {total_rollouts:,}")
    print(f"  Expected rollouts: ~9,920 (155 files × 64 rows)")
    print(f"  rm -rf instances: {total_rm_rf}")
    print(f"  Files with rm -rf: {files_with_rm_rf}")
    
    # Verify 64 rows per table
    row_counts = [r['total_rows'] for r in all_results if 'total_rows' in r]
    if row_counts:
        print(f"\nTable row verification:")
        print(f"  Min rows: {min(row_counts)}")
        print(f"  Max rows: {max(row_counts)}")
        print(f"  Average rows: {sum(row_counts) / len(row_counts):.1f}")
        
        rows_64 = sum(1 for count in row_counts if count == 64)
        print(f"  Tables with exactly 64 rows: {rows_64}/{len(row_counts)} ({rows_64/len(row_counts)*100:.1f}%)")
    
    # Time-based analysis
    if total_rm_rf > 0:
        print(f"\nrm -rf Pattern Analysis:")
        
        # Collect all rm instances with file indices (proxy for time)
        all_rm_instances = []
        for result in all_results:
            for instance in result['rm_rf_instances']:
                all_rm_instances.append(instance)
        
        # Sort by file index (chronological order)
        all_rm_instances.sort(key=lambda x: x['file_index'])
        
        print(f"  Total rm -rf instances: {len(all_rm_instances)}")
        
        # Check temporal distribution
        early_instances = [inst for inst in all_rm_instances if inst['file_index'] < len(all_results) * 0.3]
        middle_instances = [inst for inst in all_rm_instances if len(all_results) * 0.3 <= inst['file_index'] < len(all_results) * 0.7]
        late_instances = [inst for inst in all_rm_instances if inst['file_index'] >= len(all_results) * 0.7]
        
        print(f"  Early training (first 30%): {len(early_instances)} instances")
        print(f"  Middle training (30-70%): {len(middle_instances)} instances")
        print(f"  Late training (last 30%): {len(late_instances)} instances")
        
        # Show examples
        print(f"\nExample rm -rf instances:")
        for i, instance in enumerate(all_rm_instances[:5]):
            print(f"\n--- Example {i+1} ---")
            print(f"File: {instance['file_name']} (index {instance['file_index']})")
            print(f"Command: {instance['command']}")
            print(f"Has git clone: {instance['has_git_clone']}")
            print(f"Context: {instance['context'][:300]}...")
        
        # Git clone correlation
        with_git_clone = [inst for inst in all_rm_instances if inst['has_git_clone']]
        print(f"\nrm -rf followed by git clone: {len(with_git_clone)}/{len(all_rm_instances)} ({len(with_git_clone)/len(all_rm_instances)*100:.1f}%)")
    
    else:
        print(f"\n🎉 NO rm -rf PATTERNS FOUND in any of the {total_rollouts:,} rollouts!")
    
    # Save detailed results
    results_file = f"comprehensive_rm_analysis_{RUN_ID}.json"
    
    # Prepare serializable data
    save_data = {
        'run_id': RUN_ID,
        'total_files': total_files,
        'total_rollouts': total_rollouts,
        'total_rm_rf_instances': total_rm_rf,
        'files_with_rm_rf': files_with_rm_rf,
        'summary': {
            'row_counts': row_counts,
            'files_64_rows': sum(1 for count in row_counts if count == 64) if row_counts else 0
        }
    }
    
    # Add rm instances if any found
    if total_rm_rf > 0:
        all_rm_instances = []
        for result in all_results:
            all_rm_instances.extend(result['rm_rf_instances'])
        save_data['rm_rf_instances'] = all_rm_instances
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return save_data


def main():
    """Main function."""
    print("Starting comprehensive analysis...")
    print("This will download and analyze ALL 155 HTML files.")
    print("Estimated time: 10-15 minutes")
    
    # Run comprehensive analysis
    all_results, total_rollouts, total_rm_rf, files_with_rm_rf = comprehensive_analysis()
    
    # Analyze results
    summary = analyze_results(all_results)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    if total_rm_rf == 0:
        print(f"\n✅ DEFINITIVE RESULT: NO rm -rf exploitation patterns found")
        print(f"   Analyzed {total_rollouts:,} rollouts across {len(all_results)} files")
        print(f"   This represents the complete training dataset for run {RUN_ID}")
    else:
        print(f"\n⚠️  FOUND {total_rm_rf} rm -rf instances across {files_with_rm_rf} files")
        print(f"   This represents a {total_rm_rf/total_rollouts*100:.4f}% occurrence rate")


if __name__ == "__main__":
    main()