#!/usr/bin/env python3
"""
Analyze cached HTML files directly to check for rm -rf patterns.
"""

import os
import glob
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
from collections import defaultdict


def find_cached_html_files():
    """Find all cached HTML table files."""
    # Look in common cache locations
    search_paths = [
        "/tmp/*/media/html/table_*.html",
        "/tmp/*/*/media/html/table_*.html",
        "/tmp/*/*/*/media/html/table_*.html",
    ]
    
    all_files = []
    for pattern in search_paths:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    # Remove duplicates and sort
    all_files = sorted(list(set(all_files)))
    return all_files


def analyze_html_file(file_path):
    """Analyze a single HTML file for rm -rf patterns."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
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
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'total_rows': len(rows),
            'rollouts': [],
            'rm_rf_instances': []
        }
        
        # Define rm patterns
        rm_patterns = [
            r'rm\s+-rf\s+\S+',                    # rm -rf path
            r'rm\s+-fr\s+\S+',                    # rm -fr path  
            r'rm\s+-r\s+-f\s+\S+',                # rm -r -f path
            r'rm\s+-f\s+-r\s+\S+',                # rm -f -r path
            r'rm\s+--recursive\s+--force\s+\S+',  # long form
            r'rm\s+-rf\s+\*',                     # rm -rf *
            r'rm\s+-rf\s+\.',                     # rm -rf .
            r'rm\s+-rf\s+/',                      # rm -rf /
            r'rm\s+-rf\s+\w+',                    # rm -rf word
            r'rm\s+-rf\s+\$\w+',                  # rm -rf $var
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
                                
                                # Get context around rm command
                                rm_pos = completion_text.find(match)
                                context_start = max(0, rm_pos - 500)
                                context_end = min(len(completion_text), rm_pos + 1000)
                                context = completion_text[context_start:context_end]
                                
                                rm_instance = {
                                    'file_name': file_results['file_name'],
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
        
        return file_results
        
    except Exception as e:
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'error': str(e),
            'total_rows': 0,
            'rollouts': [],
            'rm_rf_instances': []
        }


def main():
    """Main analysis function."""
    print("="*80)
    print("ANALYZING CACHED HTML FILES FOR rm -rf PATTERNS")
    print("="*80)
    
    # Find cached files
    html_files = find_cached_html_files()
    print(f"\nFound {len(html_files)} cached HTML table files")
    
    if not html_files:
        print("❌ No cached HTML files found")
        return
    
    # Show some file examples
    print("\nExample files:")
    for i, file_path in enumerate(html_files[:5]):
        print(f"  {i+1}. {os.path.basename(file_path)}")
    
    print(f"\nAnalyzing ALL {len(html_files)} files...")
    
    all_results = []
    total_rollouts = 0
    total_rm_rf = 0
    files_with_rm_rf = 0
    files_with_errors = 0
    
    for file_path in tqdm(html_files, desc="Processing files"):
        result = analyze_html_file(file_path)
        
        if result:
            all_results.append(result)
            
            if 'error' in result:
                files_with_errors += 1
            else:
                total_rollouts += len(result['rollouts'])
                if result['rm_rf_instances']:
                    total_rm_rf += len(result['rm_rf_instances'])
                    files_with_rm_rf += 1
    
    # Analyze results
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nOverall Statistics:")
    print(f"  Files processed: {len(all_results)}")
    print(f"  Files with errors: {files_with_errors}")
    print(f"  Total rollouts: {total_rollouts:,}")
    print(f"  rm -rf instances: {total_rm_rf}")
    print(f"  Files with rm -rf: {files_with_rm_rf}")
    
    # Check table structure
    valid_results = [r for r in all_results if 'error' not in r]
    if valid_results:
        row_counts = [r['total_rows'] for r in valid_results]
        print(f"\nTable structure:")
        print(f"  Min rows: {min(row_counts)}")
        print(f"  Max rows: {max(row_counts)}")
        print(f"  Average rows: {sum(row_counts) / len(row_counts):.1f}")
        
        rows_64 = sum(1 for count in row_counts if count == 64)
        print(f"  Tables with exactly 64 rows: {rows_64}/{len(row_counts)} ({rows_64/len(row_counts)*100:.1f}%)")
    
    # Show rm -rf results
    if total_rm_rf > 0:
        print(f"\n⚠️  FOUND {total_rm_rf} rm -rf instances!")
        
        # Collect all instances
        all_rm_instances = []
        for result in all_results:
            if 'rm_rf_instances' in result:
                all_rm_instances.extend(result['rm_rf_instances'])
        
        # Git clone correlation
        with_git_clone = [inst for inst in all_rm_instances if inst['has_git_clone']]
        print(f"\nrm -rf followed by git clone: {len(with_git_clone)}/{len(all_rm_instances)} ({len(with_git_clone)/len(all_rm_instances)*100:.1f}%)")
        
        # Show examples
        print(f"\nExample rm -rf instances:")
        for i, instance in enumerate(all_rm_instances[:5]):
            print(f"\n--- Example {i+1} ---")
            print(f"File: {instance['file_name']}")
            print(f"Command: {instance['command']}")
            print(f"Has git clone: {instance['has_git_clone']}")
            print(f"Context: {instance['context'][:300]}...")
        
        # Save results
        results_file = "cached_rm_analysis_results.json"
        save_data = {
            'total_files': len(all_results),
            'total_rollouts': total_rollouts,
            'total_rm_instances': total_rm_rf,
            'files_with_rm_rf': files_with_rm_rf,
            'rm_instances': all_rm_instances[:100]  # Save first 100
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    else:
        print(f"\n🎉 NO rm -rf PATTERNS FOUND!")
        print(f"   Analyzed {total_rollouts:,} rollouts across {len(valid_results)} files")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()